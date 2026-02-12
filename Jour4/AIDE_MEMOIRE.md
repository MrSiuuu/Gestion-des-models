# AIDE-MÉMOIRE TECHNIQUE - TP4
## Détection de Fraude Bancaire

---

## FEATURE ENGINEERING

### Transformations Temporelles

```python
# Extraire composantes temporelles
df['hour'] = (df['Time'] / 3600) % 24
df['day'] = df['Time'] // (3600 * 24)

# Encodage cyclique (préserve la continuité)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Créer des périodes
def get_period(hour):
 if 0 <= hour < 6:
 return 'night'
 elif 6 <= hour < 12:
 return 'morning'
 elif 12 <= hour < 18:
 return 'afternoon'
 else:
 return 'evening'

df['period'] = df['hour'].apply(get_period)
```

### Transformations sur Montants

```python
# Log transformation (gérer asymétrie)
df['amount_log'] = np.log1p(df['Amount']) # log(1 + x)

# Binning
df['amount_bin'] = pd.cut(
 df['Amount'], 
 bins=[0, 10, 50, 100, 500, np.inf],
 labels=['micro', 'small', 'medium', 'large', 'xlarge']
)

# Écart à la médiane
median_amount = df['Amount'].median()
df['amount_deviation'] = df['Amount'] - median_amount

# Indicateur montant nul
df['is_zero_amount'] = (df['Amount'] == 0).astype(int)
```

### Features d'Interaction

```python
# Produits de features
df['V1_Amount'] = df['V1'] * df['Amount']
df['V2_Amount'] = df['V2'] * df['Amount']

# Ratios (attention division par zéro)
df['V1_V2_ratio'] = df['V1'] / (df['V2'] + 1e-5)

# Sommes/Moyennes
df['V_sum_top5'] = df[['V1', 'V2', 'V3', 'V4', 'V5']].sum(axis=1)
df['V_mean_top5'] = df[['V1', 'V2', 'V3', 'V4', 'V5']].mean(axis=1)
```

### Agrégations (Attention Leakage !)

```python
# Statistiques roulantes (sur données TRIÉES par Time)
df = df.sort_values('Time')

# Montant moyen des N dernières transactions
df['amount_rolling_mean'] = df['Amount'].rolling(window=10, min_periods=1).mean()

# Écart-type roulant
df['amount_rolling_std'] = df['Amount'].rolling(window=10, min_periods=1).std()

# Nombre de transactions dans la dernière heure
df['tx_last_hour'] = df.groupby(df['Time'] // 3600).cumcount()
```

---

## GESTION DU DÉSÉQUILIBRE

### Class Weight

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Automatique
model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')

# Manuel
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
 'balanced',
 classes=np.unique(y_train),
 y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
```

### Rééchantillonnage

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Oversampling (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combiné
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

---

## VALIDATION CROISÉE

### StratifiedKFold

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Préserve le ratio des classes
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
scores = cross_val_score(
 model, X_train, y_train,
 cv=skf,
 scoring='roc_auc',
 n_jobs=-1
)

print(f"ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

# Pour données temporelles
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
 X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
 y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
 
 # Entraîner et évaluer
 model.fit(X_train_fold, y_train_fold)
 score = model.score(X_val_fold, y_val_fold)
```

---

## MÉTRIQUES POUR DONNÉES DÉSÉQUILIBRÉES

### Matrice de Confusion

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculer
cm = confusion_matrix(y_test, y_pred)

# Visualiser
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Légitime', 'Fraude'])
disp.plot(cmap='Blues')
plt.title('Matrice de Confusion')
plt.show()

# Extraire TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
```

### Métriques Principales

```python
from sklearn.metrics import (
 classification_report,
 precision_score,
 recall_score,
 f1_score,
 roc_auc_score,
 average_precision_score
)

# Rapport complet
print(classification_report(y_test, y_pred, target_names=['Légitime', 'Fraude']))

# Métriques individuelles
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# AUC (nécessite probabilités)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba) # Plus pertinent !

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
```

### Courbes ROC et Precision-Recall

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Courbe ROC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC')
plt.legend()

# Courbe Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision-Recall')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## PIPELINES SCIKIT-LEARN

### Pipeline Simple

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
 ('scaler', StandardScaler()),
 ('classifier', RandomForestClassifier(random_state=42))
])

# Utilisation
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Pipeline avec ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Définir les colonnes
numeric_features = ['V1', 'V2', 'Amount', 'Time']
categorical_features = ['period']

# Transformations par type
preprocessor = ColumnTransformer(
 transformers=[
 ('num', StandardScaler(), numeric_features),
 ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
 ]
)

# Pipeline complet
pipeline = Pipeline([
 ('preprocessor', preprocessor),
 ('classifier', RandomForestClassifier(random_state=42))
])
```

### Transformer Custom

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
 def __init__(self):
 pass
 
 def fit(self, X, y=None):
 return self
 
 def transform(self, X):
 X = X.copy()
 # Vos transformations
 X['amount_log'] = np.log1p(X['Amount'])
 X['hour'] = (X['Time'] / 3600) % 24
 return X

# Utilisation dans un pipeline
pipeline = Pipeline([
 ('feature_eng', FeatureEngineer()),
 ('scaler', StandardScaler()),
 ('classifier', RandomForestClassifier())
])
```

---

## GRID SEARCH

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Définir la grille
param_grid = {
 'classifier__n_estimators': [100, 200],
 'classifier__max_depth': [10, 20, None],
 'classifier__min_samples_split': [2, 5],
 'classifier__class_weight': ['balanced']
}

# Configuration
grid_search = GridSearchCV(
 pipeline,
 param_grid,
 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
 scoring='average_precision', # PR-AUC
 n_jobs=-1,
 verbose=2
)

# Exécution
grid_search.fit(X_train, y_train)

# Résultats
print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score (CV): {grid_search.best_score_:.4f}")

# Meilleur modèle (déjà réentraîné sur tout le train)
best_model = grid_search.best_estimator_
```

### RandomizedSearchCV (Plus Rapide)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Distributions de paramètres
param_distributions = {
 'classifier__n_estimators': randint(100, 500),
 'classifier__max_depth': [10, 20, 30, None],
 'classifier__min_samples_split': randint(2, 20),
 'classifier__min_samples_leaf': randint(1, 10),
 'classifier__max_features': ['sqrt', 'log2', None]
}

# Recherche aléatoire (50 combinaisons)
random_search = RandomizedSearchCV(
 pipeline,
 param_distributions,
 n_iter=50,
 cv=5,
 scoring='average_precision',
 n_jobs=-1,
 random_state=42,
 verbose=2
)

random_search.fit(X_train, y_train)
```

---

## ANALYSE DE PERFORMANCE

### Feature Importance

```python
import pandas as pd

# Extraire l'importance
feature_names = X_train.columns
importances = best_model.named_steps['classifier'].feature_importances_

# DataFrame
feature_importance_df = pd.DataFrame({
 'feature': feature_names,
 'importance': importances
}).sort_values('importance', ascending=False)

# Visualiser Top 15
plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Features les Plus Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
 pipeline,
 X_train, y_train,
 cv=5,
 scoring='average_precision',
 train_sizes=np.linspace(0.1, 1.0, 10),
 n_jobs=-1
)

# Moyennes et écarts-types
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Score Train', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, label='Score Validation', marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel('Taille du Training Set')
plt.ylabel('Score (PR-AUC)')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

### Optimisation du Seuil

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Calculer precision/recall pour tous les seuils
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Calculer F1 pour chaque seuil
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

# Trouver le seuil optimal
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Seuil optimal: {optimal_threshold:.4f}")
print(f"F1-Score maximal: {f1_scores[optimal_idx]:.4f}")

# Prédire avec le nouveau seuil
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# Comparer
print("\nAvec seuil 0.5:")
print(classification_report(y_test, y_pred))
print("\nAvec seuil optimal:")
print(classification_report(y_test, y_pred_optimal))
```

---

## SÉRIALISATION

### Sauvegarder un Modèle

```python
import joblib

# Sauvegarder
joblib.dump(best_model, 'models/fraud_detector_v1.joblib')

# Sauvegarder avec compression
joblib.dump(best_model, 'models/fraud_detector_v1.joblib', compress=3)

# Sauvegarder métadonnées
metadata = {
 'model_version': '1.0',
 'train_date': '2025-12-14',
 'best_params': grid_search.best_params_,
 'cv_score': grid_search.best_score_,
 'threshold': optimal_threshold,
 'features': X_train.columns.tolist()
}

joblib.dump(metadata, 'models/metadata_v1.joblib')
```

### Charger un Modèle

```python
# Charger
model = joblib.load('models/fraud_detector_v1.joblib')
metadata = joblib.load('models/metadata_v1.joblib')

# Prédire
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)[:, 1]
```

---

## SNIPPETS UTILES

### Afficher les Résultats de GridSearch

```python
# Convertir en DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Top 10 configurations
top_results = results_df.sort_values('rank_test_score').head(10)
print(top_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
```

### Analyse des Erreurs

```python
# Extraire les erreurs
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Faux Positifs
fp_mask = (y_test == 0) & (y_pred == 1)
false_positives = X_test[fp_mask]
print(f"Nombre de faux positifs: {fp_mask.sum()}")

# Faux Négatifs
fn_mask = (y_test == 1) & (y_pred == 0)
false_negatives = X_test[fn_mask]
print(f"Nombre de faux négatifs: {fn_mask.sum()}")

# Analyser les caractéristiques
print("\nStatistiques des Faux Positifs:")
print(false_positives.describe())

print("\nStatistiques des Faux Négatifs:")
print(false_negatives.describe())
```

### Reproductibilité

```python
import random
import numpy as np

# Fixer tous les seeds
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Dans tous les modèles
model = RandomForestClassifier(random_state=RANDOM_STATE)

# Dans tous les splits
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

---

## RESSOURCES RAPIDES

### Imports Essentiels

```python
# Data
import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métriques
from sklearn.metrics import (
 classification_report,
 confusion_matrix,
 roc_auc_score,
 average_precision_score,
 precision_recall_curve,
 roc_curve
)

# Déséquilibre
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Utilitaires
import joblib
from tqdm import tqdm
```

---

**Bon courage ! **

