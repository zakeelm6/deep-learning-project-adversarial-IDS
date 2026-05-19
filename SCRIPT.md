# Script de présentation — 5 minutes
## Adversarially Robust Intrusion Detection System

> **Format :** 7 slides · ~5 minutes · chaque section = ce que tu dis + ce que tu dois comprendre + questions possibles

---

## SLIDE 1 — Titre *(0:00 → 0:10)*

**Ce que tu dis :**
> "Bonjour, je m'appelle Zakariya El Mansouri. Je vais vous présenter notre projet sur la robustesse adversariale des systèmes de détection d'intrusion, encadré par Pr. Tarik Fissaa."

**→ Transition vers slide 2 :**
> *"Commençons par le problème qu'on cherche à résoudre et les données sur lesquelles on travaille."*

---

## SLIDE 2 — Problème & Dataset *(0:10 → 0:55)*

**Ce que tu dis :**
> "Le problème qu'on adresse : un IDS — Intrusion Detection System — utilise du deep learning pour surveiller le trafic réseau et détecter les attaques. Mais il existe une vulnérabilité : un attaquant peut **modifier légèrement** les statistiques de son trafic — la durée de connexion, le nombre de bytes envoyés — pour tromper le modèle, sans que son attaque cesse de fonctionner.
>
> On travaille sur le dataset UNSW-NB15 : 82 000 connexions réseau avec 42 features chacune, labellisées normal ou attaque.
>
> Notre attaquant opère en **white-box** : il connaît le modèle. Il peut modifier certaines features — comme la durée ou le débit — mais pas l'adresse IP ou le protocole, sinon ça brise son attaque."

**→ Transition vers slide 3 :**
> *"Pour détecter ces attaques, on a entraîné un MLP — voyons son architecture et ses performances de départ."*

---

### Ce que tu dois comprendre

**Pourquoi white-box ?**
C'est le pire cas pour le défenseur. Si on arrive à se défendre contre un attaquant qui connaît tout, on est robuste contre n'importe qui.

**Pourquoi certaines features sont non-manipulables ?**
Un attaquant réseau réel ne peut pas changer son IP source sans briser la connexion TCP. Il ne peut pas changer de protocole sans changer d'attaque. Ce sont des contraintes physiques du réseau, pas des contraintes arbitraires.

**C'est quoi la norme L∞ et ε ?**
ε est le "budget" de l'attaquant. `||δ||∞ ≤ ε` signifie que chaque feature ne peut être modifiée de plus de ε (par exemple 0.1 après normalisation). C'est une borne pour que la perturbation reste petite et indétectable.

---

### Questions possibles

**Q : Pourquoi utiliser UNSW-NB15 et pas un autre dataset ?**
> Parce qu'il est bien documenté, utilisé dans la littérature scientifique sur les IDS adversariaux, et les features sont réalistes — elles correspondent à de vraies statistiques de connexions réseau capturées en laboratoire.

**Q : Qu'est-ce qu'une feature réseau concrètement ?**
> Par exemple `dur` = durée de la connexion en secondes, `sbytes` = nombre de bytes envoyés par la source, `sttl` = le TTL (Time To Live) du paquet source. Ce sont des statistiques agrégées sur une connexion complète.

---

## SLIDE 3 — Modèle Baseline *(0:55 → 1:45)*

**Ce que tu dis :**
> "Notre modèle de base est un MLP — Multi-Layer Perceptron — avec 4 couches. En entrée les 42 features, ensuite des couches cachées de 128, 64 et 32 neurones avec des activations ReLU et du Dropout à 30%, et en sortie une Sigmoid qui donne une probabilité entre 0 et 1. Si c'est supérieur à 0.5 : attaque. Sinon : normal.
>
> On l'entraîne avec Adam et la Binary Cross-Entropy sur 30 epochs.
>
> Sur données normales, il obtient un F1 de **0.911** — ce qui est très bon. Mais déjà **15.5% des attaques passent inaperçues** — c'est notre point de départ avant d'appliquer les attaques adversariales."

**→ Transition vers slide 4 :**
> *"Maintenant, voyons ce qu'un attaquant peut faire contre ce modèle avec FGSM et PGD."*

---

### Ce que tu dois comprendre

**Pourquoi ReLU et pas Sigmoid dans les couches cachées ?**
ReLU (`max(0,z)`) évite le problème du *vanishing gradient* : avec Sigmoid ou Tanh, le gradient devient quasi-nul pour les grandes valeurs et le réseau n'apprend plus. ReLU reste actif tant que z > 0.

**Pourquoi Dropout ?**
À chaque forward pass pendant l'entraînement, on éteint aléatoirement 30% des neurones. Ça force le réseau à ne pas "mémoriser" des chemins spécifiques et à apprendre des représentations plus robustes — c'est une forme de régularisation.

**Pourquoi F1 et pas accuracy ?**
F1 = moyenne harmonique de la Précision et du Rappel. Pour un IDS, le Rappel est critique : manquer une vraie attaque (faux négatif) est catastrophique. L'accuracy serait trompeuse car un modèle qui dit toujours "attaque" aurait 55% d'accuracy sans rien apprendre d'utile.

**C'est quoi l'Evasion Rate ?**
C'est le pourcentage des vraies attaques que le modèle classe comme "normal". C'est `1 - Recall`. Un evasion rate de 15.5% veut dire que 15.5% des attaques passent sans être détectées.

---

### Questions possibles

**Q : Pourquoi cette architecture et pas une autre ?**
> C'est un choix classique pour les données tabulaires. Les CNN sont faits pour les images (structure spatiale), les RNN pour les séquences temporelles. Pour des features indépendantes comme nos 42 colonnes réseau, le MLP est le choix standard et il fonctionne bien.

**Q : Pourquoi Binary Cross-Entropy ?**
> Parce que c'est une classification binaire. La BCE pénalise fortement les prédictions confiantes mais fausses : si le modèle dit "90% chance que ce soit normal" pour une vraie attaque, la perte est très élevée — ce qui force le modèle à être précis.

**Q : C'est quoi Adam ?**
> Adam est un optimiseur qui adapte le learning rate pour chaque paramètre individuellement. Il combine le momentum (utilise l'historique des gradients pour éviter les oscillations) et RMSProp (adapte le pas selon la magnitude des gradients passés). C'est le choix par défaut en pratique.

---

## SLIDE 4 — Attaques *(1:45 → 2:45)*

**Ce que tu dis :**
> "On applique deux attaques adversariales. La première, **FGSM**, est simple : on calcule le gradient de la perte par rapport aux features d'entrée — ce gradient nous dit dans quelle direction modifier chaque feature pour que le modèle se trompe davantage. On perturbe de ε dans cette direction, uniquement sur les features manipulables.
>
> La deuxième, **PGD**, est plus puissante : c'est FGSM répété 40 fois en petits pas, avec une projection à chaque étape pour rester dans le budget ε. C'est l'attaque de référence dans la littérature.
>
> Résultat : sous PGD, le F1 tombe de 0.911 à **0.840** et l'evasion rate monte à **26.8%** — 1 attaque sur 4 passe inaperçue."

**→ Transition vers slide 5 :**
> *"Face à ces attaques, on a testé trois défenses — avec des résultats très différents."*

---

### Ce que tu dois comprendre

**Pourquoi le gradient par rapport aux entrées et pas aux poids ?**
Normalement en deep learning, on calcule le gradient par rapport aux poids pour les mettre à jour. Ici, on fait l'inverse : on fixe les poids et on calcule le gradient par rapport à x pour trouver comment modifier l'entrée. C'est la même opération mathématique (backprop), juste appliquée différemment.

**Intuition de FGSM :**
Le signe du gradient nous dit : "si j'augmente cette feature de 0.001, la loss monte de combien ?" On prend juste le signe (+1 ou -1) et on multiplie par ε. Résultat : chaque feature manipulable est poussée exactement de ε dans la direction qui maximise l'erreur.

**Pourquoi PGD est plus fort ?**
FGSM fait un grand pas aveugle depuis le point x original. PGD recalcule le gradient à chaque étape depuis le point courant — il s'adapte et explore mieux l'espace autour de x. En 40 itérations, il trouve presque toujours un exemple adversarial plus efficace que FGSM.

**Pourquoi la Précision ne baisse presque pas mais le Recall oui ?**
Les perturbations poussent les vraies attaques vers la zone "normal" du modèle. Elles ne créent pas de faux positifs (trafic normal classé attaque). Donc la Précision reste haute, mais le Recall chute — les attaques s'évadent.

---

### Questions possibles

**Q : C'est quoi la projection dans PGD ?**
> Après chaque pas, on vérifie que la perturbation δ = x_adv - x reste dans la boule L∞ de rayon ε. Si δ_i > ε pour une feature i, on la ramène à ε. C'est un simple clamp : `δ = clip(δ, -ε, +ε)`. Ça garantit que la perturbation totale ne dépasse jamais le budget.

**Q : Pourquoi ne pas perturber toutes les features ?**
> Parce qu'un attaquant réel ne peut pas. On applique un masque binaire m qui met à 0 la perturbation sur les features non-manipulables. C'est ce qui rend notre threat model réaliste plutôt que purement théorique.

**Q : Qu'est-ce que ε = 0.1 représente concrètement ?**
> Après normalisation StandardScaler, toutes les features ont moyenne 0 et écart-type 1. ε = 0.1 correspond à une perturbation de 10% d'un écart-type — une modification légère, difficile à détecter, mais suffisante pour tromper le modèle.

---

## SLIDE 5 — Défenses & Résultats *(2:45 → 4:00)*

**Ce que tu dis :**
> "On a testé trois défenses.
>
> La première : **l'adversarial training classique** — on réentraîne le modèle sur un mix de données propres et adversariales FGSM. Ça échoue — l'evasion rate monte à 36%. Problème : on s'entraîne avec FGSM mais l'attaquant utilise PGD — robustness gap.
>
> La deuxième : **le Feature Squeezing** — on quantifie les features sur 4 bits pour effacer les petites perturbations. Evasion rate : 24.5% sous PGD. Simple mais limité.
>
> La troisième — **TRADES + PGD-7 + ratio dynamique** : TRADES change la loss — au lieu de forcer la bonne classification des adversariaux, on force les prédictions à rester similaires entre x propre et x adversarial. On entraîne avec PGD-7, même famille que l'attaquant, et on monte progressivement la proportion de 20% à 50%. Evasion **17.4%**, F1 de **0.919**.
>
> Et notre meilleure configuration : **TRADES + Feature Squeezing en pipeline**. On applique la quantification 4 bits sur l'entrée avant de passer au modèle TRADES. FS efface une partie de la perturbation, TRADES gère le reste. Résultat : evasion **10.7%** sous PGD-40 — quasi-identique aux données propres à 10.3%."

**→ Transition vers slide 6 :**
> *"Je vais maintenant vous donner les conclusions et les perspectives qu'on peut tirer de ces résultats."*

---

### Ce que tu dois comprendre

**Pourquoi l'adversarial training classique échoue ?**
- **Robustness gap :** on entraîne contre FGSM (1 étape) mais on évalue contre PGD-40 (40 étapes). Sparring trop facile → vrai combat trop dur.
- L'adversarial training pousse la Sigmoid vers des valeurs extrêmes → gradient masking.

**Pourquoi Feature Squeezing fonctionne ?**
L'attaquant calcule δ avec précision float32 (ex: +0.04731...). Après quantification sur 16 niveaux, arrondi à 0.0625 ou 0.0 — la perturbation disparaît partiellement.

**Pourquoi TRADES + PGD-7 résout les problèmes de l'AT classique ?**

Trois fixes simultanés :

| Problème | Fix appliqué | Comment ça marche |
|----------|-------------|-------------------|
| Robustness gap | PGD-7 pendant l'entraînement | Même famille que PGD-40 → pas d'écart entre entraînement et évaluation |
| Gradient masking | TRADES loss (KL) | Force des sorties lisses → Sigmoid ne sature plus → gradients informatifs |
| Oubli catastrophique | Ratio dynamique : 20% → 50% | Le modèle apprend d'abord sur données propres, puis intègre progressivement les adversariaux |

**La formule TRADES :**
```
L_total = BCE(f(x), y)  +  β × KL( f(x) || f(x_adv) )
           ↑                    ↑
    perte classique       force f(x) ≈ f(x_adv)
    (apprendre correctement)  (lisser la frontière de décision)
```
β = 6.0 (valeur standard de Zhang et al. 2019)

Différence clé : au lieu de forcer `f(x_adv) = y` (bonne classification des adversariaux), on force `f(x_adv) ≈ f(x)` (prédiction similaire). La Sigmoid ne sature plus → gradients préservés → PGD-40 ne trouve plus de "zones non couvertes".

---

**Comment fonctionne le pipeline TRADES + Feature Squeezing ?**

```
x (entrée brute)
    ↓
[Feature Squeezing — 4 bits]   ← efface les petites perturbations
    ↓
x_sq (entrée quantifiée)
    ↓
[Modèle TRADES]                ← frontière de décision lisse
    ↓
ŷ (prédiction)
```

**Pourquoi c'est orthogonal :**
- FS agit sur l'**entrée** — indépendamment du modèle. Si δ_i < 1/16 ≈ 0.0625, la feature i revient à sa valeur originale après quantification. L'attaquant a dépensé son budget sur une perturbation effacée.
- TRADES agit sur le **modèle** — il rend la frontière de décision moins sensible aux petites variations. Même si une perturbation résiduelle survit au FS, le modèle TRADES la gère mieux.

**Résultat clé :** l'écart entre evasion propres (10.3%) et sous PGD-40 (10.7%) est de seulement **0.4 point**. L'attaquant PGD-40 ne gagne quasi-rien par rapport au cas sans attaque — le pipeline neutralise presque entièrement l'attaque.

**Résultats obtenus :**

| Modèle | F1 Propres | F1 PGD-40 | Evasion PGD-40 |
|--------|-----------|-----------|----------------|
| Baseline | 0.911 | 0.839 | 26.9% |
| Adv. Training (FGSM-AT) | 0.897 | 0.770 | 36.7% ❌ |
| Feature Squeezing | 0.882 | 0.845 | 24.5% |
| TRADES + PGD-7 | 0.919 | 0.894 | 17.4% ✅ |
| **TRADES + FS (pipeline)** | **0.909** | **0.908** | **10.7%** ✅✅ |

**Point clé :** TRADES+FS a une evasion quasi-identique sur données propres (10.3%) et sous PGD-40 (10.7%) — le pipeline est pratiquement insensible à l'attaque.

**Perspectives :**
| Option | Evasion PGD-40 | F1 propres | Comment |
|--------|---------------|-----------|---------|
| TRADES + seuil 0.40 | 14.7% | 0.9245 | Juste changer le seuil, sans réentraînement |
| TRADES + FS + seuil 0.45 | 7.8% | 0.9133 | Combinaison des deux leviers |

---

### Questions possibles

**Q : C'est quoi le gradient masking ?**
> Quand le modèle est très confiant (Sigmoid ≈ 0 ou 1), le gradient de la loss est quasi-nul. PGD ne trouve plus de direction de perturbation — il semble que le modèle soit robuste. Mais c'est faux : avec plus d'itérations (PGD-40), on sort de la zone de gradient faible et on trouve des vulnérabilités que l'entraînement n'a pas couvertes.

**Q : Pourquoi 4 bits pour le Feature Squeezing ?**
> C'est un hyperparamètre. 4 bits = 16 niveaux. Avec moins de bits, on perd trop de performance sur données propres. Avec plus de bits, les perturbations survivent à la quantification. 4 bits est le meilleur compromis sur ce dataset.

**Q : Peut-on combiner Feature Squeezing et TRADES ?**
> Oui, et c'est une perspective directe. Feature Squeezing agit au niveau de l'entrée (prétraitement), TRADES agit au niveau du modèle. Les deux sont orthogonaux : on peut appliquer la quantification 4 bits avant de passer au modèle TRADES. En théorie, cela cumulerait les deux effets : réduction des perturbations + frontière de décision lisse.

---

## SLIDE 6 — Conclusion *(4:00 → 4:45)*

**Ce que tu dis :**
> "En résumé : notre MLP baseline est bon (F1 = 0.911), mais FGSM et PGD font passer 27% des attaques inaperçues. L'adversarial training classique aggrave la situation. Le Feature Squeezing seul donne 24.5%. TRADES + PGD-7 descend à 17.4%.
>
> Notre meilleur résultat implémenté : **TRADES + Feature Squeezing en pipeline** — evasion **10.7%** sous PGD-40, F1 de **0.909** sur données propres. La clé : les deux défenses sont orthogonales — FS agit sur l'entrée, TRADES agit sur le modèle.
>
> En perspective, deux pistes pour aller encore plus loin : ajuster le seuil de décision à 0.40 pour descendre à **14.7%** sans aucun réentraînement, ou combiner pipeline + seuil 0.45 pour atteindre **7.8%** — quasi-élimination des évasions adversariales. Ces résultats sont limités par le temps et les ressources disponibles — avec plus de puissance de calcul, on consoliderait encore davantage."

**→ Transition vers slide 7 :**
> *"Voilà pour la présentation — merci de votre attention."*

---

### Ce que tu dois comprendre

**Trade-off robustesse / précision (Tsipras et al. 2019) :**
Il existe une tension fondamentale prouvée mathématiquement : un modèle plus robuste est légèrement moins précis sur données propres. Ici : baseline F1=0.911 vs Feature Squeezing F1=0.882 (-3%). C'est le coût de la défense.

**C'est quoi AutoAttack ?**
Une attaque standardisée qui combine plusieurs méthodes (PGD + attaques sans gradient + attaques aléatoires) pour évaluer la robustesse de façon fiable. C'est le standard actuel de l'évaluation adversariale.

**C'est quoi la randomized smoothing ?**
Une technique qui ajoute du bruit gaussien à l'entrée et classe par vote majoritaire. Elle donne une **garantie certifiée** : "le modèle ne changera jamais de prédiction si la perturbation reste dans un rayon r". C'est de la robustesse prouvable, pas juste empirique.

---

### Questions possibles

**Q : Vos résultats se généralisent à d'autres datasets ?**
> Probablement pas directement. UNSW-NB15 a ses propres caractéristiques. Il faudrait tester sur KDD Cup 99 ou CIC-IDS-2017 pour confirmer. Mais les mécanismes d'échec de l'adversarial training (gradient masking, oubli catastrophique) sont des phénomènes généraux.

**Q : Un attaquant réel utiliserait-il FGSM/PGD ?**
> Pas directement — en pratique il n'a pas accès aux gradients. Mais des attaques black-box (estimées par requêtes multiples) ou par transfert (entraîner un modèle substitut puis attaquer) donnent des résultats proches en pratique. White-box est le worst case qui borne la vulnérabilité.

**Q : Pourquoi Feature Squeezing ne réduit-il pas l'evasion à 0 ?**
> Parce que pour ε = 0.1, certaines perturbations sont plus grandes que le niveau de quantification (1/16 ≈ 0.0625). Ces perturbations survivent partiellement à l'arrondi. Pour les éliminer totalement, il faudrait encore moins de bits — mais ça dégraderait trop les performances sur données propres.

---

## SLIDE 7 — Merci *(4:45 → 5:00)*

**Ce que tu dis :**
> "Merci. Je suis disponible pour vos questions."

---

## Résumé — les chiffres clés à retenir

| Chiffre | Signification |
|---------|--------------|
| **0.911** | F1 du baseline sur données propres |
| **26.9%** | Evasion rate sous PGD-40 (baseline) — 1 attaque sur 4 passe |
| **36.7%** | Evasion rate adv. training classique — la défense empire les choses |
| **24.5%** | Evasion rate Feature Squeezing sous PGD |
| **17.4%** | Evasion rate TRADES+PGD-7 |
| **10.7%** | Evasion rate TRADES+FS pipeline — notre meilleur résultat |
| **0.909** | F1 de TRADES+FS sur données propres |

---

## Ce qu'on a fait — récap de toutes les approches

| Approche | Ce que c'est | Résultat | Pourquoi |
|----------|-------------|---------|---------|
| Baseline MLP | 4 couches, ReLU, Dropout 0.3, Adam | F1=0.911, evasion=15.5% | Point de départ |
| FGSM attack | 1 pas dans le sens du gradient | evasion→24.8% | Attaque rapide |
| PGD-40 attack | 40 itérations + projection L∞ | evasion→26.9% | Attaque de référence |
| Adv. Training FGSM-AT | Réentraîner sur FGSM | evasion=36.7% ❌ | Robustness gap |
| Feature Squeezing seul | Quantification 4 bits à l'inférence | evasion=24.5% | Efface la précision numérique |
| TRADES + PGD-7 | Loss KL + PGD-7 + ratio 20%→50% | evasion=17.4% ✅ | Lisse la frontière de décision |
| **TRADES + FS pipeline** | FS en prétraitement + modèle TRADES | evasion=**10.7%** ✅✅ | Défenses orthogonales empilées |

---

## Les 3 phrases qui résument tout

1. **Le problème :** Un MLP bien entraîné se fait tromper par de petites perturbations des features réseau — 27% des attaques passent sous PGD-40.

2. **L'échec et la limite :** L'adversarial training classique échoue (robustness gap + gradient masking). Feature Squeezing seul reste limité à 24.5%.

3. **La solution :** Pipeline TRADES+FS — FS efface les perturbations (orthogonal au modèle), TRADES lisse la frontière de décision (KL). Résultat : **10.7% d'evasion sous PGD-40**, quasi-identique aux 10.3% sur données propres — pratiquement insensible à l'attaque.
