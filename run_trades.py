"""
TRADES + PGD-7 + Dynamic Ratio Adversarial Training
Fixes:
  1. Robustness gap   → train with PGD-7 (same family as PGD-40 evaluation)
  2. Catastrophic forgetting → dynamic ratio clean/adv (80/20 → 50/50)
  3. Gradient masking → TRADES loss (smooth outputs, no overconfidence)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(tr, te):
    for df in [tr, te]:
        df.drop(columns=[c for c in ['id','attack_cat'] if c in df.columns], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    med = tr.median(numeric_only=True)
    tr.fillna(med, inplace=True); te.fillna(med, inplace=True)
    cat = [c for c in tr.select_dtypes(include=['object','str']).columns if c != 'label']
    le = LabelEncoder()
    for col in cat:
        tr[col] = le.fit_transform(tr[col].astype(str))
        kn = set(le.classes_)
        te[col] = le.transform(te[col].astype(str).apply(lambda x: x if x in kn else le.classes_[0]))
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr.drop('label', axis=1).values.astype(np.float32))
    ytr = tr['label'].values.astype(np.float32)
    Xte = sc.transform(te.drop('label', axis=1).values.astype(np.float32))
    yte = te['label'].values.astype(np.float32)
    return Xtr, ytr, Xte, yte

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

print("=== Chargement donnees ===")
tr = pd.read_csv('data/UNSW_NB15_training-set.csv')
te = pd.read_csv('data/UNSW_NB15_testing-set.csv')
Xtr, ytr, Xte, yte = preprocess(tr.copy(), te.copy())
print(f"Train: {Xtr.shape} | Test: {Xte.shape}")

# Masque features manipulables
cols = pd.read_csv('data/UNSW_NB15_training-set.csv').drop(columns=['id','attack_cat','label']).columns.tolist()
MANIP = ['dur','spkts','dpkts','sbytes','dbytes','sttl','sload','dload','sinpkt','dinpkt','sjit','djit','smean','dmean']
midx  = [cols.index(f) for f in MANIP if f in cols]
mask  = torch.zeros(len(cols)); mask[midx] = 1.0

EPS   = 0.1
ALPHA = EPS / 7   # pas par iteration

# ── Attaques (evaluation) ──────────────────────────────────────────────────────
def fgsm(m, X, y, eps):
    Xt = torch.FloatTensor(X).requires_grad_(True)
    nn.BCELoss()(m(Xt).squeeze(), torch.FloatTensor(y)).backward()
    return (Xt + eps * Xt.grad.sign() * mask).detach().numpy()

def pgd(m, X, y, eps, steps=40):
    Xo = torch.FloatTensor(X); Xd = Xo.clone(); yt = torch.FloatTensor(y)
    a  = eps / steps
    for _ in range(steps):
        Xd = Xd.detach().requires_grad_(True)
        nn.BCELoss()(m(Xd).squeeze(), yt).backward()
        Xd = Xo + torch.clamp(Xd + a * Xd.grad.sign() * mask - Xo, -eps, eps)
    return Xd.detach().numpy()

def fsq(X, b=4):
    mv = np.max(np.abs(X), axis=0, keepdims=True) + 1e-8
    return np.round(X / mv * (2**b)) / (2**b) * mv

def evl(m, X, y):
    m.eval()
    with torch.no_grad():
        prob = m(torch.FloatTensor(X)).squeeze().numpy()
        pred = (prob >= 0.5).astype(int)
    return {
        'F1': f1_score(y, pred), 'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred), 'PR-AUC': average_precision_score(y, prob),
        'Evasion%': (pred[y==1] == 0).mean() * 100
    }

# ── PGD-7 pour l'entraînement (TRADES adversarial generation) ─────────────────
def pgd_train(m, X_nat, eps, steps=7):
    """Génère x_adv en maximisant KL(f(x_nat) || f(x_adv)) — version TRADES"""
    m.eval()
    X_nat = X_nat.detach()
    X_adv = X_nat + 0.001 * torch.randn_like(X_nat)  # init aléatoire dans la boule
    a = eps / steps
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        p_nat = m(X_nat).squeeze().detach()
        p_adv = m(X_adv).squeeze()
        # KL(p_nat || p_adv) pour distribution de Bernoulli
        kl = (p_nat * torch.log((p_nat + 1e-8) / (p_adv + 1e-8)) +
              (1 - p_nat) * torch.log((1 - p_nat + 1e-8) / (1 - p_adv + 1e-8))).mean()
        kl.backward()
        X_adv = X_nat + torch.clamp(
            X_adv + a * X_adv.grad.sign() * mask - X_nat, -eps, eps
        )
    return X_adv.detach()

# ── TRADES Loss ───────────────────────────────────────────────────────────────
def trades_loss(m, X_nat, y, X_adv, beta=6.0):
    """
    L_TRADES = BCE(f(x), y)  +  beta * KL(f(x) || f(x_adv))
    BCE sur données propres  +  terme de régularisation de lissage
    """
    p_nat = m(X_nat).squeeze()
    p_adv = m(X_adv).squeeze()
    loss_ce = F.binary_cross_entropy(p_nat, y)
    loss_kl = (p_nat.detach() * torch.log((p_nat.detach() + 1e-8) / (p_adv + 1e-8)) +
               (1 - p_nat.detach()) * torch.log((1 - p_nat.detach() + 1e-8) / (1 - p_adv + 1e-8))).mean()
    return loss_ce + beta * loss_kl

# ── Entraînement TRADES ───────────────────────────────────────────────────────
print("\n=== TRADES + PGD-7 + Ratio Dynamique ===")
mt = MLP(Xtr.shape[1])
opt = torch.optim.Adam(mt.parameters(), lr=0.001)

EPOCHS = 20
BATCH  = 1024
BETA   = 6.0  # coefficient TRADES standard (Zhang et al. 2019)

Xtr_t = torch.FloatTensor(Xtr)
ytr_t = torch.FloatTensor(ytr)

for ep in range(EPOCHS):
    mt.train()

    # Ratio dynamique : commence 80% propres / 20% adversariaux → finit 50/50
    adv_ratio = 0.20 + 0.30 * (ep / (EPOCHS - 1))  # 0.20 → 0.50
    n_adv = int(len(Xtr) * adv_ratio)

    # Sélectionner les exemples qui recevront une perturbation
    adv_idx = np.random.choice(len(Xtr), n_adv, replace=False)
    X_batch_nat = Xtr_t[adv_idx]

    # Générer les exemples adversariaux avec PGD-7 (TRADES)
    X_batch_adv = pgd_train(mt, X_batch_nat, EPS, steps=7)

    # Shuffle complet du batch d'entraînement
    idx = np.random.permutation(len(Xtr))
    total_loss = 0.0

    mt.train()
    for s in range(0, len(idx), BATCH):
        b_idx = idx[s:s+BATCH]
        Xb    = Xtr_t[b_idx]
        yb    = ytr_t[b_idx]

        # Quels indices du batch ont un équivalent adversarial ?
        adv_mask_b = np.isin(b_idx, adv_idx)
        if adv_mask_b.any():
            # Reconstruire x_adv pour ce mini-batch
            local_adv_idx = np.where(adv_mask_b)[0]
            # Map b_idx → adv_idx position
            global_adv_pos = np.searchsorted(np.sort(adv_idx), b_idx[adv_mask_b])
            Xb_adv = Xb.clone()
            Xb_adv[local_adv_idx] = X_batch_adv[global_adv_pos % len(X_batch_adv)]
        else:
            Xb_adv = Xb.clone()

        opt.zero_grad()
        loss = trades_loss(mt, Xb, yb, Xb_adv, beta=BETA)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    n_batches = max(1, len(idx) // BATCH)
    print(f"  Epoch {ep+1:02d}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | Adv ratio: {adv_ratio:.0%}")

torch.save(mt.state_dict(), 'results/trades_model.pth')
print("trades_model.pth sauvegarde")

# ── Chargement baseline ───────────────────────────────────────────────────────
print("\n=== Chargement Baseline ===")
mb = MLP(Xtr.shape[1])
mb.load_state_dict(torch.load('results/baseline_model.pth', weights_only=True))
mb.eval()

# ── Génération des attaques (sur sous-ensemble attaques seulement) ─────────────
print("\n=== Generation attaques (PGD-40) ===")
aidx = np.where(yte == 1)[0]
Xa = Xte[aidx]; ya = yte[aidx]

print("  FGSM baseline...")
Xfb = Xte.copy(); Xfb[aidx] = fgsm(mb, Xa, ya, EPS)
print("  PGD-40 baseline...")
Xpb = Xte.copy(); Xpb[aidx] = pgd(mb, Xa, ya, EPS, steps=40)
print("  PGD-40 TRADES...")
Xpt = Xte.copy(); Xpt[aidx] = pgd(mt, Xa, ya, EPS, steps=40)
print("  Feature Squeezing...")
Xsq = fsq(Xte); Xpsq = fsq(Xpb)
print("Attaques OK")

# ── Évaluation ────────────────────────────────────────────────────────────────
print("\n=== RESULTATS FINAUX ===")
rows = [
    ('Baseline',        'Propres', evl(mb, Xte,  yte)),
    ('Baseline',        'FGSM',    evl(mb, Xfb,  yte)),
    ('Baseline',        'PGD-40',  evl(mb, Xpb,  yte)),
    ('TRADES+PGD-7',    'Propres', evl(mt, Xte,  yte)),
    ('TRADES+PGD-7',    'PGD-40',  evl(mt, Xpt,  yte)),
    ('Feat.Squeezing',  'Propres', evl(mb, Xsq,  yte)),
    ('Feat.Squeezing',  'PGD-40',  evl(mb, Xpsq, yte)),
]

print(f"\n{'Modele':<18} {'Scenario':<10} {'F1':>7} {'Recall':>8} {'Evasion%':>10}")
print("-" * 58)
for m, s, r in rows:
    print(f"{m:<18} {s:<10} {r['F1']:>7.4f} {r['Recall']:>8.4f} {r['Evasion%']:>9.1f}%")

pd.DataFrame([{
    'Modele': m, 'Scenario': s,
    'F1': r['F1'], 'Precision': r['Precision'],
    'Recall': r['Recall'], 'PR-AUC': r['PR-AUC'], 'Evasion%': r['Evasion%']
} for m, s, r in rows]).to_csv('results/trades_results.csv', index=False)
print("\ntrades_results.csv sauvegarde")

# ── Graphique comparatif ───────────────────────────────────────────────────────
print("\n=== Graphique ===")
models   = ['Baseline', 'TRADES+PGD-7', 'Feat.Squeezing']
colors   = ['#0A3064', '#00A5D7', '#2ecc71']
sc_clean = 'Propres'
sc_att   = 'PGD-40'

f1_clean, f1_att, ev_clean, ev_att = [], [], [], []
for mn in models:
    rc = next((r for m,s,r in rows if m==mn and s==sc_clean), None)
    ra = next((r for m,s,r in rows if m==mn and s==sc_att),   None)
    if rc is None: rc = next((r for m,s,r in rows if m==mn), None)
    if ra is None: ra = rc
    f1_clean.append(rc['F1']);      f1_att.append(ra['F1'])
    ev_clean.append(rc['Evasion%']); ev_att.append(ra['Evasion%'])

x = np.arange(len(models)); w = 0.3
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))

b1 = a1.bar(x - w/2, f1_clean, w, label='Données propres', color=[c+'99' for c in colors])
b2 = a1.bar(x + w/2, f1_att,   w, label='Sous PGD-40',     color=colors)
a1.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
a1.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)
a1.set_title('F1-score : Propres vs Sous PGD-40', fontweight='bold')
a1.set_xticks(x); a1.set_xticklabels(models); a1.legend(); a1.set_ylim(0, 1.1)
a1.axhline(0.9, color='gray', linestyle='--', alpha=0.4, label='F1=0.9')

b3 = a2.bar(x - w/2, ev_clean, w, label='Données propres', color=[c+'99' for c in colors])
b4 = a2.bar(x + w/2, ev_att,   w, label='Sous PGD-40',     color=colors)
a2.bar_label(b3, fmt='%.1f%%', padding=3, fontsize=9)
a2.bar_label(b4, fmt='%.1f%%', padding=3, fontsize=9)
a2.set_title('Evasion Rate : Propres vs Sous PGD-40', fontweight='bold')
a2.set_xticks(x); a2.set_xticklabels(models); a2.legend()

plt.suptitle('Baseline vs TRADES+PGD-7 vs Feature Squeezing', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/trades_comparison.png', dpi=150)
plt.close()
print("trades_comparison.png sauvegarde")
print("\n=== DONE ===")
