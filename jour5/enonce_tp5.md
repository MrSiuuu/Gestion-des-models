# TP5 - Techniques Avancées & MLOps

**Master 1 Data Engineering - YNOV Montpellier**  
**Cours 5 : Techniques Avancées & MLOps**  
**Durée : 4 heures**  
**Enseignant : BENHAMDI Ayoub**

---

## 📋 Contexte & Objectifs

### Contexte Métier

Vous travaillez pour un opérateur de télécommunications qui souhaite protéger ses clients contre les SMS indésirables (spam). Votre mission est de développer un système de détection automatique de spam SMS basé sur l'analyse du contenu textuel des messages.

Le défi principal : **environ 13.4% des messages sont du spam**, ce qui représente un problème de classes déséquilibrées nécessitant des techniques avancées de Machine Learning et de traitement du langage naturel (NLP).

### Objectifs Pédagogiques

À l'issue de ce TP, vous serez capable de :

1. ✅ **Traiter des données textuelles** avec vectorisation TF-IDF
2. ✅ **Maîtriser les algorithmes avancés** : SVM, KNN, XGBoost et LightGBM
3. ✅ **Optimiser les hyperparamètres** avec RandomizedSearchCV
4. ✅ **Réduire la dimensionnalité** avec TruncatedSVD et feature selection
5. ✅ **Appliquer les concepts MLOps** : versioning, tracking et monitoring
6. ✅ **Gérer le déséquilibre de classes** avec des techniques appropriées

---

## 🎯 Compétences Visées

- **NLP (Natural Language Processing)** : Vectorisation TF-IDF, analyse de texte
- **Algorithmique ML** : Implémenter et comparer des algorithmes sophistiqués
- **Optimisation** : Trouver les meilleurs hyperparamètres efficacement
- **Évaluation** : Utiliser des métriques adaptées (F1-Score, ROC-AUC, Precision-Recall)
- **MLOps** : Tracker les expérimentations avec MLflow
- **Production** : Créer des pipelines robustes et reproductibles

---

## 📦 Prérequis

### Installation

```bash
# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset Kaggle (voir data/README.md)
kaggle datasets download -d uciml/sms-spam-collection-dataset
```

### Dataset

- **Source** : Kaggle - SMS Spam Collection Dataset
- **Lien** : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Taille** : ~500 KB (5,574 messages SMS)
- **Spam** : ~747 (13.4%)
- **Ham (légitime)** : ~4,827 (86.6%)
- **Langue** : Anglais

⚠️ **Important** : Téléchargez le dataset **AVANT** le début du TP (voir `data/README.md` pour les instructions détaillées).

### Fichiers fournis

- `utils.py` : Fonctions utilitaires pour les visualisations
- `tp5_template.py` : Script Python à compléter
- `data/README.md` : Instructions pour télécharger le dataset

---

## 🏗️ Structure du TP

Le TP est divisé en **4 parties progressives** :

### **Partie 1** : Exploration et Préparation (45 min) - `/20`
- Chargement et analyse exploratoire
- Analyse des caractéristiques textuelles
- Vectorisation TF-IDF
- Split train/test stratifié

### **Partie 2** : Algorithmes Avancés (90 min) - `/30`
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost et LightGBM
- Comparaison des performances

### **Partie 3** : Optimisation & Dimensionnalité (60 min) - `/30`
- RandomizedSearchCV
- Réduction de dimension avec TruncatedSVD
- Feature selection
- Learning curves

### **Partie 4** : Introduction MLOps (45 min) - `/20`
- MLflow tracking
- Pipelines de production
- Monitoring et data drift
- Sauvegarde et versioning

---

## 📝 Partie 1 : Exploration et Préparation (45 min)

### Objectifs

- Comprendre la structure et les caractéristiques du dataset
- Analyser les messages spam vs ham
- Vectoriser le texte avec TF-IDF
- Préparer les données pour la modélisation

### Tâches à réaliser

#### 1.1 Chargement des données

```python
# Utiliser la fonction fournie dans utils.py
from utils import load_spam_dataset

df = load_spam_dataset('data/spam.csv')
```

#### 1.2 Analyse Exploratoire des Données (EDA)

- Afficher les dimensions, types de données et statistiques descriptives
- Vérifier les valeurs manquantes
- Analyser la répartition de la variable cible `label` (spam/ham)
- Calculer le ratio de déséquilibre
- **Analyser la longueur des messages** :
  - Nombre de caractères par message
  - Nombre de mots par message
  - Comparer spam vs ham

**Questions de réflexion** :
- Quel est le pourcentage de spam ?
- Pourquoi ce déséquilibre pose-t-il problème pour les algorithmes ML classiques ?
- Les messages spam sont-ils généralement plus longs ou plus courts que les messages légitimes ?

#### 1.3 Visualisations

Créer les visualisations suivantes :

1. **Distribution de la variable cible** (bar plot spam vs ham)
2. **Distribution de la longueur des messages** :
   - Histogramme du nombre de caractères (spam vs ham)
   - Histogramme du nombre de mots (spam vs ham)
3. **Top 15-20 mots les plus fréquents** :
   - Dans les messages spam
   - Dans les messages ham
   - Comparer les différences
4. **(Optionnel)** Nuage de mots (WordCloud) pour spam et ham

💡 **Astuce** : Utilisez `seaborn` et `matplotlib` pour des graphiques professionnels.

**Code exemple pour analyser les mots fréquents** :

```python
from collections import Counter
import re

def get_word_frequency(messages, top_n=20):
    # Concaténer tous les messages
    text = ' '.join(messages)
    # Extraire les mots (minuscules, sans ponctuation)
    words = re.findall(r'\b\w+\b', text.lower())
    # Compter les occurrences
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

# Exemple d'utilisation
spam_messages = df[df['label'] == 'spam']['message']
top_spam_words = get_word_frequency(spam_messages, top_n=20)
```

#### 1.4 Préparation des données

**Étape 1 : Encoder la cible**

```python
# Convertir 'spam' → 1 et 'ham' → 0
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['label'])  # spam=1, ham=0
```

**Étape 2 : Vectorisation TF-IDF**

Le texte brut ne peut pas être utilisé directement par les algorithmes ML. Il faut le convertir en vecteurs numériques avec **TF-IDF** (Term Frequency-Inverse Document Frequency).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Créer le vectoriseur
vectorizer = TfidfVectorizer(
    max_features=3000,        # Garder les 3000 mots les plus importants
    stop_words='english',      # Supprimer les mots courants (the, is, at...)
    lowercase=True,            # Tout en minuscules
    ngram_range=(1, 2)        # Utiliser uni-grams et bi-grams
)

# Vectoriser les messages
X = vectorizer.fit_transform(df['message'])

print(f"Shape de X: {X.shape}")  # (5574, 3000)
print(f"Type de X: {type(X)}")   # sparse matrix (efficace en mémoire)
```

💡 **Explications** :
- **TF-IDF** : Mesure l'importance d'un mot dans un document par rapport à tous les documents
- **max_features** : Limite le vocabulaire (sinon trop de features)
- **stop_words** : Mots courants sans valeur sémantique
- **ngram_range=(1,2)** : Capture les mots seuls (uni-grams) et les paires de mots (bi-grams)

**Étape 3 : Split train/test stratifié**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,           # CRUCIAL : préserver les proportions
    random_state=42
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nDistribution train: {pd.Series(y_train).value_counts()}")
print(f"Distribution test: {pd.Series(y_test).value_counts()}")
```

⚠️ **ATTENTION** : 
- **NE PAS** faire `fit_transform()` sur le test set → DATA LEAKAGE !
- Toujours `fit()` sur train, puis `transform()` sur test
- Avec le pipeline (Partie 4), cela sera automatique

**Livrable Partie 1** :
- ✅ Dataset chargé et analysé
- ✅ Au moins 3 visualisations pertinentes
- ✅ Texte vectorisé avec TF-IDF (3000 features)
- ✅ Données préparées (X_train, X_test, y_train, y_test)

---

## 🤖 Partie 2 : Algorithmes Avancés (90 min)

### Objectifs

- Implémenter et comparer 4 algorithmes sophistiqués
- Comprendre l'impact des hyperparamètres
- Évaluer avec des métriques adaptées au déséquilibre

### 2.1 Support Vector Machine (SVM)

#### Implémentation

```python
from sklearn.svm import SVC
from time import time

# TODO: Créer et entraîner un SVM
svm_model = SVC(
    kernel='rbf',              # Noyau RBF (Gaussian)
    C=1.0,                     # Régularisation
    class_weight='balanced',   # CRUCIAL pour le déséquilibre
    probability=True,          # Pour predict_proba (ROC-AUC)
    random_state=42
)

start_time = time()
svm_model.fit(X_train, y_train)
svm_time = time() - start_time

print(f"✓ SVM entraîné en {svm_time:.2f}s")
```

#### Évaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Prédictions
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]  # Probabilités classe positive

# Métriques
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_proba_svm)

print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1-Score: {svm_f1:.4f}")
print(f"ROC-AUC: {svm_auc:.4f}")
```

#### Visualisations

```python
from utils import plot_confusion_matrix, plot_roc_curve

# Matrice de confusion
plot_confusion_matrix(y_test, y_pred_svm, title="SVM RBF - Matrice de Confusion")

# Courbe ROC
plot_roc_curve(y_test, y_proba_svm, model_name="SVM RBF")
```

#### Expérimentation : Tester différents noyaux

Testez **3 configurations** :

1. **SVM Linear** : `kernel='linear'`, `C=1.0`
2. **SVM RBF** : `kernel='rbf'`, `C=1.0` (déjà fait)
3. **SVM Polynomial** : `kernel='poly'`, `degree=3`, `C=1.0`

Pour chaque configuration, calculez et comparez les métriques.

💡 **Note** : Le noyau **linear** est souvent très performant pour les données textuelles en haute dimension (TF-IDF crée ~3000 features).

**Question 2.1** : Quel noyau performe le mieux pour la classification de texte ? Pourquoi ?

_Réponse attendue_ : Linear est généralement excellent pour le texte car les données TF-IDF sont déjà en haute dimension et souvent linéairement séparables.

---

### 2.2 K-Nearest Neighbors (KNN)

#### Implémentation

```python
from sklearn.neighbors import KNeighborsClassifier

# TODO: Tester différentes valeurs de K
k_values = [3, 5, 10, 20, 50]
knn_scores = []

print("Test de différentes valeurs de K:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    knn_scores.append(f1)
    print(f"  K={k}: F1={f1:.4f}")
```

#### Visualisation

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel('Nombre de voisins (K)', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Performance KNN en fonction de K', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()
```

#### Évaluation du meilleur K

```python
# TODO: Identifier le meilleur K
best_k_idx = np.argmax(knn_scores)
best_k = k_values[best_k_idx]

print(f"\nMeilleur K: {best_k}")

# Entraîner et évaluer complètement
knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

knn_f1 = f1_score(y_test, y_pred_knn)
knn_auc = roc_auc_score(y_test, y_proba_knn)

print(f"F1-Score: {knn_f1:.4f}")
print(f"ROC-AUC: {knn_auc:.4f}")
```

**Question 2.2** : Pourquoi un K trop petit ou trop grand est problématique ?

_Réponse_ : K trop petit = overfitting (sensible au bruit), K trop grand = underfitting (frontière trop lisse).

**Question 2.3** : KNN est-il adapté pour la classification de texte en haute dimension ? Justifiez.

_Réponse_ : KNN n'est pas idéal car :
- Très lent en prédiction (calcul de distances pour tous les points)
- "Curse of dimensionality" avec 3000 features
- Distances moins significatives en haute dimension

---

### 2.3 XGBoost

#### Calcul du scale_pos_weight

```python
# Pour gérer le déséquilibre de classes
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")
```

#### Implémentation

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # CRUCIAL
    random_state=42,
    eval_metric='logloss'
)

start_time = time()
xgb_model.fit(X_train, y_train)
xgb_time = time() - start_time

print(f"✓ XGBoost entraîné en {xgb_time:.2f}s")
```

#### Évaluation

```python
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)

print(f"F1-Score: {xgb_f1:.4f}")
print(f"ROC-AUC: {xgb_auc:.4f}")
```

#### Analyse des features importantes

```python
# Récupérer les noms des mots (features)
feature_names = vectorizer.get_feature_names_out()
importances = xgb_model.feature_importances_

# Top 20 mots les plus importants
top_indices = np.argsort(importances)[::-1][:20]

print("\nTop 20 mots les plus importants pour détecter le spam:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
```

💡 **Interprétation** : Ces mots sont ceux qui permettent le mieux de distinguer spam vs ham. Vérifiez qu'ils ont du sens (ex: "free", "win", "prize", "call" pour spam).

---

### 2.4 LightGBM

```python
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

start_time = time()
lgbm_model.fit(X_train, y_train)
lgbm_time = time() - start_time

y_pred_lgbm = lgbm_model.predict(X_test)
y_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

lgbm_f1 = f1_score(y_test, y_pred_lgbm)
lgbm_auc = roc_auc_score(y_test, y_proba_lgbm)

print(f"✓ LightGBM entraîné en {lgbm_time:.2f}s")
print(f"F1-Score: {lgbm_f1:.4f}")
print(f"ROC-AUC: {lgbm_auc:.4f}")
print(f"\nComparaison vitesse: XGBoost={xgb_time:.2f}s vs LightGBM={lgbm_time:.2f}s")
```

---

### 2.5 Comparaison Globale

```python
# Créer un dictionnaire avec tous les résultats
results = {
    'SVM': {
        'accuracy': svm_accuracy,
        'precision': svm_precision,
        'recall': svm_recall,
        'f1': svm_f1,
        'auc': svm_auc
    },
    'KNN': {
        'f1': knn_f1,
        'auc': knn_auc
        # TODO: Ajouter les autres métriques
    },
    'XGBoost': {
        'f1': xgb_f1,
        'auc': xgb_auc
        # TODO: Compléter
    },
    'LightGBM': {
        'f1': lgbm_f1,
        'auc': lgbm_auc
        # TODO: Compléter
    }
}

# Afficher la comparaison
from utils import compare_models_performance
compare_models_performance(results)
```

**Questions de synthèse** :

**Q2.4** : Quel modèle performe le mieux sur ce problème de détection de spam ? Selon quelle métrique ?

_Réponse_ : [À compléter après expérimentation]

**Q2.5** : Pourquoi l'Accuracy n'est-elle PAS une bonne métrique ici ?

_Réponse_ : Avec 86.6% de ham, un modèle stupide prédisant toujours "ham" aurait 86.6% d'accuracy mais ne détecterait aucun spam. F1-Score et ROC-AUC sont plus informatifs.

**Q2.6** : Quel est le trade-off entre Precision et Recall dans le contexte de détection de spam ?

_Réponse_ :
- **Haute Precision** : Peu de faux positifs (messages légitimes marqués spam) → Meilleure expérience utilisateur
- **Haut Recall** : Attraper tous les spam → Meilleure protection mais risque de bloquer des messages légitimes
- **Trade-off** : Dépend de la priorité métier (protection vs expérience)

**Livrable Partie 2** :
- ✅ 4 algorithmes implémentés et évalués
- ✅ Tableau comparatif complet
- ✅ Analyse des mots importants pour la détection
- ✅ Réponses aux questions de réflexion

---

## ⚙️ Partie 3 : Optimisation & Dimensionnalité (60 min)

### Objectifs

- Optimiser les hyperparamètres efficacement
- Réduire la dimensionnalité avec TruncatedSVD
- Sélectionner les features les plus pertinentes

### 3.1 RandomizedSearchCV sur XGBoost

#### Définir l'espace de recherche

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

print(f"Nombre d'hyperparamètres: {len(param_distributions)}")
print(f"Combinaisons possibles: {4*4*4*3*3*3*3:,}")
```

#### Lancer RandomizedSearchCV

```python
print("Lancement de RandomizedSearchCV (plusieurs minutes)...")

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=50,                              # 50 combinaisons testées
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',                           # Métrique à optimiser
    n_jobs=-1,                              # Parallélisation
    verbose=1,
    random_state=42
)

start_time = time()
random_search.fit(X_train, y_train)
optim_time = time() - start_time

print(f"\n✓ Optimisation terminée en {optim_time:.2f}s ({optim_time/60:.2f} min)")
```

#### Analyser les résultats

```python
print("\nMeilleurs hyperparamètres:")
for param, value in random_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nMeilleur score CV (F1): {random_search.best_score_:.4f}")
print(f"Score de base: {xgb_f1:.4f}")
print(f"Amélioration: {(random_search.best_score_ - xgb_f1)*100:+.2f}%")
```

#### Évaluer sur le test set

```python
xgb_optimized = random_search.best_estimator_

y_pred_xgb_opt = xgb_optimized.predict(X_test)
y_proba_xgb_opt = xgb_optimized.predict_proba(X_test)[:, 1]

xgb_opt_f1 = f1_score(y_test, y_pred_xgb_opt)
xgb_opt_auc = roc_auc_score(y_test, y_proba_xgb_opt)

print(f"\nPerformances sur test set:")
print(f"F1-Score optimisé: {xgb_opt_f1:.4f} (baseline: {xgb_f1:.4f})")
print(f"ROC-AUC optimisé: {xgb_opt_auc:.4f} (baseline: {xgb_auc:.4f})")

plot_confusion_matrix(y_test, y_pred_xgb_opt, title="XGBoost Optimisé")
```

**Question 3.1** : Pourquoi utiliser RandomizedSearchCV plutôt que GridSearchCV ?

_Réponse_ : RandomizedSearchCV teste N combinaisons aléatoires (50 ici) au lieu de TOUTES (6,912 ici). C'est beaucoup plus rapide avec des performances souvent similaires.

---

### 3.2 Réduction de Dimension avec TruncatedSVD

💡 **Pourquoi TruncatedSVD et pas PCA ?** 

TF-IDF produit des **matrices creuses** (beaucoup de zéros). `TruncatedSVD` est optimisé pour ce type de données, contrairement à `PCA` qui nécessite des matrices denses.

#### Application

```python
from sklearn.decomposition import TruncatedSVD

# Réduire à 100 composantes
svd = TruncatedSVD(n_components=100, random_state=42)

X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

variance_explained = svd.explained_variance_ratio_.sum()

print(f"Dimensions originales: {X_train.shape[1]}")
print(f"Dimensions après SVD: {X_train_svd.shape[1]}")
print(f"Variance expliquée: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
```

#### Visualisation

```python
cumsum_variance = np.cumsum(svd.explained_variance_ratio_)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumsum_variance)+1), cumsum_variance, marker='o', linewidth=2)
plt.xlabel('Nombre de composantes', fontsize=12)
plt.ylabel('Variance expliquée cumulée', fontsize=12)
plt.title('Variance Expliquée par TruncatedSVD', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()
```

#### Modélisation avec SVD

```python
print("Entraînement XGBoost avec TruncatedSVD...")
start_time = time()

xgb_svd = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_svd.fit(X_train_svd, y_train)
svd_time = time() - start_time

y_pred_svd = xgb_svd.predict(X_test_svd)
svd_f1 = f1_score(y_test, y_pred_svd)

print(f"\nComparaison Sans SVD vs Avec SVD:")
print(f"Sans SVD: F1={xgb_f1:.4f}, Temps={xgb_time:.2f}s, Features={X_train.shape[1]}")
print(f"Avec SVD: F1={svd_f1:.4f}, Temps={svd_time:.2f}s, Features={X_train_svd.shape[1]}")
```

**Question 3.2** : Dans quel contexte la réduction de dimensionnalité est-elle recommandée avant KNN ?

_Réponse_ : KNN souffre du "curse of dimensionality". Réduire les dimensions avec SVD/PCA améliore les performances et la vitesse.

---

### 3.3 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, chi2

# Sélectionner les K=500 meilleures features
selector = SelectKBest(score_func=chi2, k=500)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Identifier les features sélectionnées
selected_mask = selector.get_support()
selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

print(f"\nFeature Selection:")
print(f"Features originales: {X_train.shape[1]}")
print(f"Features sélectionnées: {X_train_selected.shape[1]}")
print(f"\nExemples de mots sélectionnés: {selected_features[:20]}")
```

#### Modélisation

```python
xgb_fs = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_fs.fit(X_train_selected, y_train)
y_pred_fs = xgb_fs.predict(X_test_selected)
fs_f1 = f1_score(y_test, y_pred_fs)

print(f"\nComparaison:")
print(f"Toutes features: F1={xgb_f1:.4f}")
print(f"500 features: F1={fs_f1:.4f}")
```

---

### 3.4 Learning Curves (Optionnel)

```python
from sklearn.model_selection import learning_curve

print("Calcul des learning curves...")

train_sizes, train_scores, val_scores = learning_curve(
    xgb_optimized,
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Visualisation
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_mean, label='Score Train', marker='o')
plt.plot(train_sizes, val_mean, label='Score Validation', marker='s')
plt.xlabel("Taille du set d'entraînement", fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Learning Curves - XGBoost Optimisé', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Livrable Partie 3** :
- ✅ Hyperparamètres optimaux trouvés
- ✅ TruncatedSVD appliquée et analysée
- ✅ Feature selection effectuée
- ✅ Comparaisons des performances

---

## 🚀 Partie 4 : Introduction MLOps (45 min)

### Objectifs

- Tracker les expérimentations avec MLflow
- Créer un pipeline de production
- Simuler un monitoring de data drift
- Assurer la reproductibilité

### 4.1 MLflow Tracking

#### Configuration

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("spam_detection_tp5")

print("✓ MLflow configuré")
print("Pour voir l'UI MLflow: mlflow ui (puis http://localhost:5000)")
```

#### Tracker le modèle baseline

```python
with mlflow.start_run(run_name="xgboost_baseline"):
    # Logger hyperparamètres
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    
    # Logger métriques
    mlflow.log_metric("f1_score", xgb_f1)
    mlflow.log_metric("roc_auc", xgb_auc)
    
    # Logger le modèle
    mlflow.sklearn.log_model(xgb_model, "model")
    
    print("✓ Run 'xgboost_baseline' logged")
```

#### Tracker le modèle optimisé

```python
with mlflow.start_run(run_name="xgboost_optimized"):
    # Logger tous les best_params
    for param, value in random_search.best_params_.items():
        mlflow.log_param(param, value)
    
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("optimization_method", "RandomizedSearchCV")
    
    # Logger métriques
    mlflow.log_metric("f1_score", xgb_opt_f1)
    mlflow.log_metric("roc_auc", xgb_opt_auc)
    mlflow.log_metric("f1_improvement", xgb_opt_f1 - xgb_f1)
    
    # Logger le modèle
    mlflow.sklearn.log_model(xgb_optimized, "model")
    
    print("✓ Run 'xgboost_optimized' logged")
```

---

### 4.2 Pipeline de Production

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Créer un pipeline complet : Texte → TF-IDF → Modèle
production_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ('model', xgb_optimized)
])

print("✓ Pipeline créé")
print(production_pipeline)
```

#### Entraîner sur données brutes

```python
# Récupérer les messages bruts (pas encore vectorisés)
messages = df['message'].values
labels = le.transform(df['label'])

# Split
messages_train, messages_test, y_train_raw, y_test_raw = train_test_split(
    messages, labels, test_size=0.2, stratify=labels, random_state=42
)

# Entraîner le pipeline
print("Entraînement du pipeline sur données brutes...")
production_pipeline.fit(messages_train, y_train_raw)

# Tester
y_pred_pipeline = production_pipeline.predict(messages_test)
pipeline_f1 = f1_score(y_test_raw, y_pred_pipeline)

print(f"✓ Pipeline entraîné - F1-Score: {pipeline_f1:.4f}")
```

💡 **Avantage** : Le pipeline peut prendre du texte brut en entrée !

```python
# Test avec de nouveaux messages
new_messages = [
    "Congratulations! You've won a FREE prize! Call now!",
    "Hey, are we still meeting for lunch tomorrow?"
]

predictions = production_pipeline.predict(new_messages)
for msg, pred in zip(new_messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"\n[{label}] {msg}")
```

#### Sauvegarder

```python
import joblib

joblib.dump(production_pipeline, 'spam_detector_pipeline.pkl')
print("✓ Pipeline sauvegardé: spam_detector_pipeline.pkl")

# Test de rechargement
loaded_pipeline = joblib.load('spam_detector_pipeline.pkl')
print("✓ Pipeline rechargé avec succès")
```

---

### 4.3 Monitoring - Data Drift

#### Analyser les distributions

```python
# Analyser la longueur des messages
df['message_length'] = df['message'].str.len()

train_df = df.iloc[:len(messages_train)].copy()
test_df = df.iloc[len(messages_train):].copy()

# Comparer les longueurs
print("Longueur moyenne des messages:")
print(f"Train: {train_df['message_length'].mean():.2f}")
print(f"Test: {test_df['message_length'].mean():.2f}")

# Visualiser
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(train_df['message_length'], bins=30, alpha=0.6, label='Train', density=True)
plt.hist(test_df['message_length'], bins=30, alpha=0.6, label='Test', density=True)
plt.xlabel('Longueur du message', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.title('Distribution de la longueur des messages', fontsize=14)
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([train_df['message_length'], test_df['message_length']], labels=['Train', 'Test'])
plt.ylabel('Longueur', fontsize=12)
plt.title('Boxplot longueur messages', fontsize=14)
plt.show()
```

#### Simuler un drift

```python
# Créer des messages "production" avec des caractéristiques différentes
# Ex: Nouveau type de spam (crypto, COVID, etc.)

new_spam_messages = [
    "URGENT: Buy Bitcoin now! 1000% guaranteed returns! Limited time!",
    "COVID-19 vaccine available NOW! Click here for instant access!",
    "Get rich quick with NFTs! Join our exclusive group!",
]

# Prédire
predictions = production_pipeline.predict(new_spam_messages)
probas = production_pipeline.predict_proba(new_spam_messages)

for msg, pred, proba in zip(new_spam_messages, predictions, probas):
    label = "SPAM" if pred == 1 else "HAM"
    conf = proba[pred]
    print(f"\n[{label}] (confiance: {conf:.2f})")
    print(f"{msg}")
```

💡 **En production** : Si de nombreux nouveaux messages ont des mots jamais vus à l'entraînement (ex: "NFT", "crypto"), le modèle peut moins bien performer → nécessité de ré-entraîner.

**Question 4.1** : Comment détecteriez-vous automatiquement un drift en production ?

_Réponse_ :
- Monitorer la distribution des scores de prédiction
- Comparer les mots fréquents (nouveaux mots non vus)
- Tests statistiques (KS-test, PSI)
- Tracker les métriques business (taux de spam détecté)

---

### 4.4 Versioning et Reproductibilité

```python
from utils import save_model_info

metadata = {
    'model_version': 'v1.0',
    'f1_score': float(xgb_opt_f1),
    'roc_auc': float(xgb_opt_auc),
    'hyperparameters': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v)
                        for k, v in random_search.best_params_.items()},
    'training_samples': int(len(messages_train)),
    'test_samples': int(len(messages_test)),
    'spam_ratio': float((y_train_raw == 1).sum() / len(y_train_raw)),
    'max_features': 3000,
    'ngram_range': '(1, 2)'
}

save_model_info(production_pipeline, 'model_info_v1.json', metadata=metadata)
print("✓ Métadonnées sauvegardées")
```

**Livrable Partie 4** :
- ✅ 2 runs trackées dans MLflow
- ✅ Pipeline de production créé et testé
- ✅ Analyse de drift effectuée
- ✅ Modèle sauvegardé avec métadonnées

---

## 📊 Livrables Finaux

À rendre **à la fin du TP** :

1. **Script Python complété** : `tp5_votrenam.py`
   - Code propre et commenté
   - Réponses aux questions de réflexion

2. **Pipeline sauvegardé** : `spam_detector_pipeline.pkl`

3. **Rapport MLflow** : Export ou screenshots de vos runs

4. **README personnel** :
   - Meilleur modèle et ses performances
   - Top 10-20 mots les plus discriminants pour détecter le spam
   - Difficultés rencontrées
   - Pistes d'amélioration

---

## 💡 Conseils & Bonnes Pratiques

### Pour réussir ce TP

- ✅ **Fixez random_state=42** partout pour la reproductibilité
- ✅ **TF-IDF : fit() sur train uniquement**, puis transform() sur test
- ✅ **Utilisez les pipelines** pour éviter le data leakage
- ✅ **Privilégiez F1-Score et ROC-AUC** plutôt que l'accuracy
- ✅ **Analysez les mots importants** pour comprendre le modèle
- ✅ **Commentez votre code** pour expliquer vos choix

### Pièges à éviter

- ❌ Faire `fit()` du TfidfVectorizer sur le test set (DATA LEAKAGE!)
- ❌ Ne pas utiliser `class_weight='balanced'` ou `scale_pos_weight`
- ❌ Oublier la validation croisée stratifiée
- ❌ Se fier uniquement à l'accuracy
- ❌ Ignorer l'interprétabilité (quels mots détectent le spam ?)

### Ressources utiles

- **Scikit-learn TF-IDF** : https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- **XGBoost** : https://xgboost.readthedocs.io/
- **LightGBM** : https://lightgbm.readthedocs.io/
- **MLflow** : https://mlflow.org/docs/latest/

---

## 🎓 Pour Aller Plus Loin (Bonus)

Si vous avez terminé en avance, explorez ces pistes :

1. **Word2Vec ou BERT** : Embeddings plus avancés que TF-IDF
2. **N-grams avancés** : Tri-grams, character n-grams
3. **Nettoyage NLP** : Stemming, lemmatisation avec NLTK/spaCy
4. **SMOTE** : Sur-échantillonnage de la classe minoritaire
5. **Ensemble Methods** : Stacking de plusieurs modèles
6. **SHAP Values** : Interpréter quels mots influencent chaque prédiction
7. **API REST** : Déployer avec Flask/FastAPI pour classifier des SMS en temps réel
8. **Analyse d'erreurs** : Examiner les faux positifs et faux négatifs

---

## 📞 Support

En cas de difficulté :

1. Consultez les fonctions de `utils.py`
2. Vérifiez la documentation officielle
3. Levez la main pour demander de l'aide
4. Collaborez avec vos voisins (sans copier-coller !)

---

**Bon courage et bon TP ! 🚀**

_Le NLP est passionnant : vous allez voir comment l'IA "comprend" le texte !_
