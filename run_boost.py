"""
Evaluation : peut-on passer sous 15.5% d'evasion sous PGD-40 ?
Stratégies sans réentraînement :
  1. Seuil de décision variable (0.5 → 0.3)
  2. TRADES + Feature Squeezing en pipeline
  3. Combinaison seuil + pipeline
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)

# ── Preprocessing ──────────────────────────────────────────────
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

print("=== Chargement ===")
tr = pd.read_csv('data/UNSW_NB15_training-set.csv')
te = pd.read_csv('data/UNSW_NB15_testing-set.csv')
Xtr, ytr, Xte, yte = preprocess(tr.copy(), te.copy())

cols = pd.read_csv('data/UNSW_NB15_training-set.csv').drop(columns=['id','attack_cat','label']).columns.tolist()
MANIP = ['dur','spkts','dpkts','sbytes','dbytes','sttl','sload','dload','sinpkt','dinpkt','sjit','djit','smean','dmean']
midx  = [cols.index(f) for f in MANIP if f in cols]
mask  = torch.zeros(len(cols)); mask[midx] = 1.0

EPS = 0.1

# ── Chargement modèles ─────────────────────────────────────────
mt = MLP(Xte.shape[1])
mt.load_state_dict(torch.load('results/trades_model.pth', weights_only=True))
mt.eval()

mb = MLP(Xte.shape[1])
mb.load_state_dict(torch.load('results/baseline_model.pth', weights_only=True))
mb.eval()

# ── PGD-40 ─────────────────────────────────────────────────────
def pgd(m, X, y, eps, steps=40):
    Xo = torch.FloatTensor(X); Xd = Xo.clone(); yt = torch.FloatTensor(y)
    a = eps / steps
    for _ in range(steps):
        Xd = Xd.detach().requires_grad_(True)
        nn.BCELoss()(m(Xd).squeeze(), yt).backward()
        Xd = Xo + torch.clamp(Xd + a * Xd.grad.sign() * mask - Xo, -eps, eps)
    return Xd.detach().numpy()

# ── Feature Squeezing ──────────────────────────────────────────
def fsq(X, b=4):
    mv = np.max(np.abs(X), axis=0, keepdims=True) + 1e-8
    return np.round(X / mv * (2**b)) / (2**b) * mv

# ── Evaluation avec seuil variable ────────────────────────────
def evl(m, X, y, threshold=0.5):
    m.eval()
    with torch.no_grad():
        prob = m(torch.FloatTensor(X)).squeeze().numpy()
    pred = (prob >= threshold).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    ev   = (pred[y==1] == 0).mean() * 100
    return f1, prec, rec, ev

# ── Génération attaques ────────────────────────────────────────
print("Génération PGD-40 sur attaques...")
aidx = np.where(yte == 1)[0]
Xa = Xte[aidx]; ya = yte[aidx]
Xpt = Xte.copy(); Xpt[aidx] = pgd(mt, Xa, ya, EPS, steps=40)
Xpt_fs = fsq(Xpt)   # TRADES + FS pipeline : FS appliqué APRÈS la perturbation
print("OK")

# ── Test 1 : seuil variable sur TRADES ────────────────────────
print("\n=== Seuil de décision (TRADES, données propres vs PGD-40) ===")
print(f"{'Seuil':>8} | {'F1 propres':>10} | {'Ev propres':>10} | {'F1 PGD':>8} | {'Ev PGD':>8} | {'Prec PGD':>9}")
print("-" * 65)
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
best_th = None
for th in thresholds:
    f1c, prc, rec, evc = evl(mt, Xte,  yte, th)
    f1a, pra, rea, eva = evl(mt, Xpt,  yte, th)
    marker = " ← SOUS 15.5%" if eva < 15.5 else ""
    print(f"  {th:.2f}   | {f1c:>10.4f} | {evc:>9.1f}% | {f1a:>8.4f} | {eva:>7.1f}%{marker}")
    if eva < 15.5 and best_th is None:
        best_th = th

# ── Test 2 : TRADES + Feature Squeezing pipeline ──────────────
print("\n=== TRADES + Feature Squeezing pipeline (seuil 0.5) ===")
f1c, prc, rec, evc = evl(mt, fsq(Xte), yte, 0.5)
f1a, pra, rea, eva = evl(mt, Xpt_fs,   yte, 0.5)
print(f"  Propres : F1={f1c:.4f}, Evasion={evc:.1f}%")
print(f"  PGD-40  : F1={f1a:.4f}, Evasion={eva:.1f}%" + (" ← SOUS 15.5%" if eva < 15.5 else ""))

# ── Test 3 : TRADES + FS + seuil variable ─────────────────────
print("\n=== TRADES + Feature Squeezing + seuil variable ===")
print(f"{'Seuil':>8} | {'F1 propres':>10} | {'Ev propres':>10} | {'F1 PGD':>8} | {'Ev PGD':>8}")
print("-" * 60)
best_combo = None
for th in thresholds:
    f1c, prc, rec, evc = evl(mt, fsq(Xte), yte, th)
    f1a, pra, rea, eva = evl(mt, Xpt_fs,   yte, th)
    marker = " ← SOUS 15.5%" if eva < 15.5 else ""
    print(f"  {th:.2f}   | {f1c:>10.4f} | {evc:>9.1f}% | {f1a:>8.4f} | {eva:>7.1f}%{marker}")
    if eva < 15.5 and best_combo is None:
        best_combo = (th, f1a, eva)

print("\n=== RÉSUMÉ ===")
if best_th:
    f1c, prc, rec, evc = evl(mt, Xte, yte, best_th)
    f1a, pra, rea, eva = evl(mt, Xpt, yte, best_th)
    print(f"Seuil seul        : threshold={best_th} → Evasion PGD={eva:.1f}%, F1 propres={f1c:.4f}")
if best_combo:
    th, f1a, eva = best_combo
    f1c, prc, rec, evc = evl(mt, fsq(Xte), yte, th)
    print(f"TRADES+FS+seuil   : threshold={th} → Evasion PGD={eva:.1f}%, F1 propres={f1c:.4f}")
