# DATASET - CREDIT CARD FRAUD DETECTION

## Informations Générales

**Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
**Taille** : ~150 MB (284,807 transactions) 
**Format** : CSV 
**Période** : 2 jours de transactions (Septembre 2013)

---

## TÉLÉCHARGEMENT

### Option 1 : Script Automatique (Recommandé)

```bash
python download_data.py
```

**Prérequis** :
1. Compte Kaggle créé
2. API Kaggle configurée :
 - Aller sur https://www.kaggle.com/settings/account
 - Cliquer sur "Create New API Token"
 - Placer `kaggle.json` dans `~/.kaggle/`
 - `chmod 600 ~/.kaggle/kaggle.json`

### Option 2 : Téléchargement Manuel

1. Aller sur https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Cliquer sur "Download" (connexion requise)
3. Décompresser `creditcard.csv`
4. Placer le fichier dans ce dossier (`TD/data/`)

---

## STRUCTURE DU DATASET

### Colonnes (30 features + 1 cible)

| Colonne | Type | Description |
|---------|------|-------------|
| `Time` | float | Secondes écoulées depuis la 1ère transaction |
| `V1` à `V28` | float | Composantes PCA (anonymisées) |
| `Amount` | float | Montant de la transaction (€) |
| `Class` | int | 0 = Légitime, 1 = Fraude |

### Caractéristiques

- **Nombre de transactions** : 284,807
- **Fraudes** : 492 (0.172%)
- **Légitimes** : 284,315 (99.828%)
- **Ratio** : 1 fraude pour 577 transactions légitimes

### Anonymisation

Les features `V1` à `V28` sont des **composantes principales (PCA)** issues d'une transformation des features originales pour :
- Protéger la confidentialité des données bancaires
- Réduire la dimensionnalité
- Préserver l'information discriminante

**Seules `Time` et `Amount` ne sont pas transformées.**

---

## STATISTIQUES

### Distribution des Montants

**Légitimes** :
- Moyenne : 88.29 €
- Médiane : 22.00 €
- Écart-type : 250.11 €
- Max : 25,691.16 €

**Fraudes** :
- Moyenne : 122.21 €
- Médiane : 9.25 €
- Écart-type : 256.68 €
- Max : 2,125.87 €

### Distribution Temporelle

- **Durée totale** : 172,792 secondes (~48 heures)
- **Transactions par heure** : ~5,933
- **Pic d'activité** : Heures de journée
- **Creux** : Nuit (0h-6h)

---

## POINTS D'ATTENTION

### 1. Déséquilibre Extrême
- **0.172% de fraudes** → Problème majeur pour le ML
- **Solutions** : class_weight, SMOTE, métriques adaptées (PR-AUC)

### 2. Anonymisation
- Les features V1-V28 sont **difficiles à interpréter**
- Pas de signification métier directe
- Focus sur `Time` et `Amount` pour le Feature Engineering

### 3. Données Temporelles
- Ordre chronologique important
- Utiliser `TimeSeriesSplit` pour validation
- Attention au data leakage dans les features roulantes

### 4. Outliers
- Beaucoup d'outliers dans `Amount`
- Utiliser `RobustScaler` plutôt que `StandardScaler`

---

## EXPLORATION RAPIDE

```python
import pandas as pd

# Charger
df = pd.read_csv('creditcard.csv')

# Infos
print(df.shape)
print(df.info())
print(df['Class'].value_counts())

# Statistiques
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum()) # Aucune valeur manquante

# Distribution des classes
print(f"Fraudes: {(df['Class']==1).sum()} ({(df['Class']==1).mean()*100:.3f}%)")
```

---

## RÉFÉRENCES

### Publication Originale
- **Titre** : "Calibrating Probability with Undersampling for Unbalanced Classification"
- **Auteurs** : Dal Pozzolo et al.
- **Année** : 2015
- **Lien** : [IEEE Xplore](https://ieeexplore.ieee.org/document/7280527)

### Citation
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
Calibrating Probability with Undersampling for Unbalanced Classification. 
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
```

---

## CHECKLIST

Avant de commencer le TP, vérifiez :

- [ ] Le fichier `creditcard.csv` est présent dans ce dossier
- [ ] Le fichier fait ~150 MB
- [ ] Il contient 284,807 lignes et 31 colonnes
- [ ] Aucune valeur manquante
- [ ] 492 fraudes (0.172%)

**Commande de vérification** :
```bash
ls -lh creditcard.csv
wc -l creditcard.csv
```

---

**Bon TP ! **

