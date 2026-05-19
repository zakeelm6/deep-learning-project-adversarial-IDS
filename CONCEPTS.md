# Explication des concepts clés — Adversarially Robust IDS

> Ce fichier explique en termes simples chaque concept technique du projet,
> pour que tu puisses répondre à n'importe quelle question du professeur.

---

## 1. Les 42 features — c'est quoi ?

Une **feature** (ou caractéristique) est une information mesurable sur une connexion réseau.
Quand un ordinateur envoie des données à un autre, on peut mesurer plein de choses sur cette
connexion : combien de temps elle a duré, combien d'octets ont été envoyés, etc.

Le dataset UNSW-NB15 contient **42 features** par connexion. En voici les principales :

| Feature | Signification | Exemple de valeur |
|---------|--------------|-------------------|
| `dur` | Durée de la connexion (secondes) | 0.002 |
| `sbytes` | Bytes envoyés par la source | 1500 |
| `dbytes` | Bytes envoyés par la destination | 320 |
| `spkts` | Nombre de paquets source | 5 |
| `dpkts` | Nombre de paquets destination | 3 |
| `sttl` | Time-To-Live du paquet source | 64 |
| `dttl` | Time-To-Live du paquet destination | 128 |
| `sload` | Débit de la source (bits/sec) | 600000 |
| `dload` | Débit de la destination (bits/sec) | 128000 |
| `sinpkt` | Temps entre deux paquets source (ms) | 0.4 |
| `dinpkt` | Temps entre deux paquets destination (ms) | 0.7 |
| `sjit` | Variation (jitter) du timing source | 0.01 |
| `djit` | Variation (jitter) du timing destination | 0.02 |
| `smean` | Taille moyenne des paquets source | 300 |
| `dmean` | Taille moyenne des paquets destination | 107 |
| `proto` | Protocole utilisé (TCP, UDP...) | tcp |
| `service` | Service réseau (http, ftp...) | http |
| `state` | État de la connexion (FIN, SYN...) | FIN |

**Pourquoi 42 features ?**
Plus on a d'informations sur une connexion, plus le modèle peut distinguer
une connexion normale d'une attaque. C'est comme un médecin qui analyse
plusieurs symptômes avant de poser un diagnostic.

**Features manipulables vs non-manipulables :**
- **Manipulables** (l'attaquant peut les modifier) : `dur`, `sbytes`, `sttl`, `sload`...
  → L'attaquant peut ralentir légèrement son trafic ou changer le volume d'octets envoyés.
- **Non-manipulables** : `proto`, `srcip`, `dport`
  → Changer l'adresse IP source briserait la connexion TCP. Changer le protocole
  changerait la nature de l'attaque elle-même.

---

## 2. Les neurones et les couches cachées (128 → 64 → 32)

### C'est quoi un neurone artificiel ?

Un neurone artificiel est une **fonction mathématique simple** :
il reçoit des valeurs en entrée, les multiplie par des poids, les additionne,
et applique une fonction d'activation.

```
entrées : x1, x2, x3, ...
          ↓   ↓   ↓
poids  : w1, w2, w3, ...
          ↓
somme  : z = w1*x1 + w2*x2 + w3*x3 + biais
          ↓
sortie : activation(z)  → ReLU, Sigmoid...
```

**Analogie :** un neurone c'est comme un vote pondéré.
Chaque feature vote (avec un poids) pour dire "est-ce une attaque ?".
Le neurone agrège ces votes.

### Pourquoi des couches de 128, 64 et 32 neurones ?

Le MLP a 3 couches cachées avec de moins en moins de neurones :

```
42 features en entrée
    ↓
Couche 1 : 128 neurones   ← extraction de patterns complexes
    ↓
Couche 2 : 64 neurones    ← résumé / abstraction
    ↓
Couche 3 : 32 neurones    ← représentation compacte
    ↓
1 neurone (Sigmoid)       ← probabilité d'être une attaque (0 à 1)
```

**Pourquoi cette forme "entonnoir" ?**
- La couche large (128) peut détecter beaucoup de patterns différents dans les 42 features.
- Les couches plus petites (64, 32) forcent le réseau à **résumer** les informations importantes.
- C'est comme lire un livre (128 mots importants) → faire un résumé (64 points clés) →
  retenir l'essentiel (32 idées) → décision finale (oui/non).

### C'est quoi ReLU ?

ReLU = Rectified Linear Unit. C'est la fonction d'activation dans les couches cachées :

```
ReLU(z) = max(0, z)
```

Si z > 0, on garde la valeur. Si z < 0, on met 0.

**Pourquoi ReLU et pas autre chose ?**
- Évite le problème de **vanishing gradient** : avec Sigmoid dans les couches cachées,
  le gradient (signal d'apprentissage) disparaît progressivement dans les couches profondes.
- ReLU reste actif pour z > 0 → le gradient passe sans s'atténuer.

### C'est quoi le Dropout (0.3) ?

À chaque passage d'entraînement, on éteint **aléatoirement 30% des neurones**.
Cela force le réseau à ne pas dépendre d'un seul chemin → meilleure généralisation.
C'est comme apprendre à conduire sans regarder un seul miroir — tu apprends à utiliser tous les indices.

---

## 3. Adam, Binary Cross-Entropy et 30 epochs

### C'est quoi l'entraînement d'un réseau de neurones ?

L'entraînement = ajuster les poids w1, w2, w3... pour que le réseau fasse
de bonnes prédictions. On répète en boucle :

```
1. Propagation avant (forward pass) :
   donner les features au réseau → obtenir une prédiction ŷ

2. Calcul de la perte (loss) :
   comparer ŷ avec y (vrai label) → mesurer l'erreur

3. Rétropropagation (backward pass) :
   calculer le gradient → savoir comment modifier les poids

4. Mise à jour des poids :
   poids = poids - lr × gradient
```

### Binary Cross-Entropy (BCE) — la fonction de perte

La BCE mesure l'erreur entre la prédiction et la vraie valeur :

```
BCE = - [ y × log(ŷ) + (1-y) × log(1-ŷ) ]
```

- Si le modèle dit ŷ = 0.95 et y = 1 (vraie attaque) → perte faible ✓
- Si le modèle dit ŷ = 0.05 et y = 1 (vraie attaque) → perte élevée ✗

**Pourquoi BCE et pas l'accuracy ?**
L'accuracy (taux de bonne réponses) n'est pas dérivable — on ne peut pas calculer
son gradient. BCE est continue et dérivable → on peut optimiser.

### Adam — l'optimiseur

Adam (Adaptive Moment Estimation) ajuste automatiquement le **pas d'apprentissage** (lr)
pour chaque poids individuellement.

```
lr = 0.001  (learning rate de départ)
```

Adam combine deux techniques :
- **Momentum** : utilise la moyenne des gradients passés → évite les oscillations.
- **RMSProp** : adapte le lr selon la variance des gradients → gros pas quand le gradient est stable,
  petit pas quand il oscille.

**Résultat :** Adam converge plus vite et plus stablement que la descente de gradient simple.

### 30 epochs

Une **epoch** = le réseau a vu tous les exemples d'entraînement une fois.

Avec 82 332 exemples et 30 epochs → le réseau voit chaque connexion **30 fois**,
en apprenant un peu à chaque passage. C'est comme réviser un cours 30 fois
avant un examen.

---

## 4. C'est quoi le gradient ?

### Définition simple

Le gradient est la **dérivée** de la fonction de perte par rapport aux paramètres.
Il indique : *"si j'augmente ce poids d'un tout petit peu, la perte monte ou descend ?"*

```
gradient = dLoss / dw

Si gradient > 0 : augmenter w augmente la perte → il faut diminuer w
Si gradient < 0 : augmenter w diminue la perte → il faut augmenter w
```

**Mise à jour :**
```
w_nouveau = w_ancien - lr × gradient
```

### Gradient par rapport aux entrées (attaques adversariales)

Normalement, on calcule le gradient par rapport aux **poids** (pour les mettre à jour).

Dans les attaques adversariales, on fait l'inverse : on fixe les poids du modèle
et on calcule le gradient par rapport aux **features d'entrée x** :

```
∇_x L = dLoss / dx
```

Ce gradient indique : *"si j'augmente la feature i d'un tout petit peu,
la perte (l'erreur du modèle) monte ou descend ?"*

FGSM exploite ça directement :
```
x_adv = x + ε × sign(∇_x L)
```
On pousse chaque feature dans la direction qui maximise l'erreur → le modèle se trompe.

### C'est quoi le gradient masking ?

Le **gradient masking** est un phénomène où les gradients deviennent quasi-nuls,
rendant les attaques basées sur les gradients inefficaces — mais c'est une fausse robustesse.

Ça arrive quand la Sigmoid est poussée vers des valeurs très confiantes (0.001 ou 0.999).
Dans ces zones, la dérivée de Sigmoid est quasi-nulle :

```
Sigmoid'(z) = Sigmoid(z) × (1 - Sigmoid(z))

Si Sigmoid(z) ≈ 1 → Sigmoid'(z) ≈ 0 × 1 = quasi-nul
Si Sigmoid(z) ≈ 0 → Sigmoid'(z) ≈ 1 × 0 = quasi-nul
```

→ PGD-10 ne trouve plus de direction de perturbation (gradient ≈ 0).
→ Mais PGD-40 fait plus d'itérations et finit par sortir de la zone de gradient faible,
  trouvant des directions vulnérables que l'entraînement n'a pas couvertes.

---

## 5. Feature Squeezing — explication détaillée

### Principe

La quantification sur 4 bits signifie qu'on **arrondit** chaque feature
à l'un des 2⁴ = **16 niveaux** possibles.

```
Formule :
X_sq = round(X / X_max × 16) / 16 × X_max
```

**Exemple concret :**
```
Feature originale   :  sbytes = 0.4731  (après normalisation)
Perturbation FGSM   :  δ = +0.0312  → sbytes_adv = 0.5043
Après quantification : round(0.5043 / 1.0 × 16) / 16 = round(8.07) / 16 = 8/16 = 0.5000

La perturbation 0.5043 → arrondie à 0.5000
→ l'écart résiduel = 0.5043 - 0.5000 = 0.0043  (au lieu de 0.0312)
→ la perturbation est atténuée de 86%
```

### Pourquoi ça fonctionne

L'attaquant calcule δ avec une précision **float32** (≈ 7 chiffres significatifs).
Après quantification à 16 niveaux, la résolution devient 1/16 ≈ **0.0625**.

Toute perturbation plus petite que 0.0625 est effacée par arrondi.
Avec ε = 0.1, les perturbations FGSM (1 seul pas) font souvent < 0.0625 → effacées.

### Limites

Avec ε = 0.1, certaines perturbations PGD (40 étapes) peuvent atteindre
la magnitude maximale. Si δ_i = 0.09, après quantification :
```
0.09 / (1/16) = 1.44 → arrondi à 1 → 1 × (1/16) = 0.0625
```
La perturbation passe de 0.09 à 0.0625 — atténuée mais pas effacée.
→ Evasion résiduelle de 24.5% (vs 26.9% sans défense).

### Pourquoi 4 bits et pas 3 ou 8 ?

| Bits | Niveaux | Evasion PGD | F1 propres | Commentaire |
|------|---------|-------------|-----------|-------------|
| 3 | 8 | ~18% | ~0.860 | Perte trop élevée sur données propres |
| **4** | **16** | **24.5%** | **0.882** | Meilleur compromis |
| 8 | 256 | ~26% | ~0.910 | Trop fin → perturbations survivent |

---

## 6. TRADES — explication complète

### Le problème que TRADES résout

L'adversarial training classique force :
```
f(x_adv) = y   (bien classifier les exemples adversariaux)
```

Problème : pour forcer cette classification, le modèle pousse la Sigmoid
vers des valeurs extrêmes (0 ou 1) → gradient masking → fausse robustesse.

### L'idée de TRADES

Au lieu de forcer la bonne classification de x_adv,
TRADES force les **prédictions similaires** entre x et x_adv :

```
f(x_adv) ≈ f(x)   (prédiction stable, peu importe la perturbation)
```

La loss TRADES :
```
L_TRADES = BCE(f(x), y)  +  β × KL(f(x) || f(x_adv))
              ↑                       ↑
   apprendre correctement      forcer f(x) ≈ f(x_adv)
   sur données propres         (lisser la frontière de décision)
```

**KL divergence :** mesure à quel point deux distributions de probabilité sont différentes.
```
KL(p || q) = p × log(p/q) + (1-p) × log((1-p)/(1-q))
```
Si f(x) = f(x_adv), alors KL = 0 → aucune pénalité.
Plus les prédictions divergent, plus la pénalité est élevée.

### Pourquoi β = 6.0 ?

β contrôle l'équilibre entre classification correcte et stabilité :
- β trop faible → le modèle se concentre trop sur les données propres → peu robuste
- β trop élevé → le modèle optimise la stabilité mais mal la classification

β = 6.0 est la valeur recommandée par Zhang et al. (ICML 2019) après validation sur plusieurs datasets.

### Les 3 fixes de TRADES + PGD-7 + ratio dynamique

**Fix 1 — Robustness gap → PGD-7 :**
L'attaque d'entraînement utilise PGD-7 (même algorithme que PGD-40 de l'évaluation,
juste moins d'itérations). L'écart entre entraînement et évaluation est réduit.

**Fix 2 — Gradient masking → Loss KL :**
Le terme KL force f(x_adv) ≈ f(x) sans exiger f(x_adv) = y.
La Sigmoid n'est pas poussée vers 0/1 → gradients préservés → pas de gradient masking.

**Fix 3 — Oubli catastrophique → ratio dynamique :**
```
Epoch 1  : 20% exemples adversariaux, 80% propres
Epoch 10 : 35% adversariaux, 65% propres
Epoch 20 : 50% adversariaux, 50% propres
```
Le modèle apprend d'abord la distribution naturelle,
puis intègre progressivement les adversariaux.

### Pipeline TRADES + Feature Squeezing

Le pipeline combine les deux défenses **en séquence** :

```
x (connexion réseau)
    ↓
[Feature Squeezing — 4 bits]
   → efface les perturbations < 0.0625
   → l'attaquant a dépensé son budget sur du bruit effacé
    ↓
x_sq (entrée quantifiée et partiellement nettoyée)
    ↓
[Modèle TRADES]
   → frontière de décision lisse (KL)
   → gère les perturbations résiduelles qui ont survécu au FS
    ↓
ŷ (prédiction finale)
```

**Pourquoi la combinaison est meilleure :**

| Mécanisme | Ce qu'il fait | Limite seul |
|-----------|--------------|-------------|
| Feature Squeezing | Efface les petites perturbations | Grandes perturbations survivent (24.5%) |
| TRADES | Rend le modèle stable aux petites variations | Perturbation pas effacée à l'entrée (17.4%) |
| **TRADES + FS** | **FS efface, TRADES gère le reste** | **10.7%** ✅✅ |

**Résultat final :**
```
Baseline         sous PGD-40 : evasion = 28.2%
TRADES seul      sous PGD-40 : evasion = 17.4%  (-10.8 pts)
FS seul          sous PGD-40 : evasion = 24.5%  (-3.7 pts)
TRADES + FS      sous PGD-40 : evasion = 10.7%  (-17.5 pts)
```

L'écart entre données propres (10.3%) et sous PGD-40 (10.7%) est de **0.4 point** seulement.
L'attaquant PGD-40 ne gagne quasi-rien — le pipeline neutralise l'attaque.

---

## Résumé en une phrase par concept

| Concept | En une phrase |
|---------|--------------|
| **42 features** | Statistiques mesurées sur chaque connexion réseau (durée, volume, timing...) |
| **Neurone** | Somme pondérée des entrées + fonction d'activation = une décision simple |
| **Couches 128→64→32** | Entonnoir qui extrait, résume puis concentre l'information |
| **BCE Loss** | Mesure l'erreur de prédiction de façon dérivable pour pouvoir optimiser |
| **Adam** | Optimiseur qui adapte le pas d'apprentissage pour chaque paramètre |
| **30 epochs** | Le réseau voit les données 30 fois en apprenant progressivement |
| **Gradient** | Direction dans laquelle modifier les poids (ou les entrées) pour maximiser/minimiser la perte |
| **Gradient masking** | Gradients quasi-nuls → fausse robustesse car PGD-40 contourne |
| **Feature Squeezing** | Arrondir à 16 niveaux pour effacer les petites perturbations numériques |
| **TRADES** | Loss KL qui force des prédictions stables sans saturer la Sigmoid |
| **Pipeline TRADES+FS** | FS nettoie l'entrée + TRADES gère le résidu = deux défenses orthogonales |
