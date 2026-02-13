"""
TP5 - Techniques Avancées & MLOps
Master 1 Data Engineering - YNOV Montpellier
Sujet : Détection de Spam SMS (énoncé uniquement)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
from time import time

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import (
    load_spam_dataset,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models_performance,
    save_model_info
)

import joblib

print("✓ Imports OK")

# =============================================================================
# PARTIE 1 : EXPLORATION ET PRÉPARATION
# =============================================================================

print("\n" + "="*80)
print("PARTIE 1 : EXPLORATION ET PRÉPARATION")
print("="*80)

# 1.1 Chargement des données
df = load_spam_dataset('data/spam.csv')

# 1.2 Analyse exploratoire
print("\n--- Dimensions ---")
print(f"Shape: {df.shape}")

print("\n--- Types ---")
print(df.dtypes)

print("\n--- Statistiques descriptives ---")
print(df.describe(include='all'))

print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

print("\n--- Répartition label (spam/ham) ---")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

# Ratio de déséquilibre
n_ham = (df['label'] == 'ham').sum()
n_spam = (df['label'] == 'spam').sum()
ratio_desequilibre = n_ham / n_spam
print(f"\nRatio déséquilibre (ham/spam): {ratio_desequilibre:.2f}")

# Longueur des messages
df['nb_caracteres'] = df['message'].str.len()
df['nb_mots'] = df['message'].str.split().str.len()

print("\n--- Longueur des messages (spam vs ham) ---")
print(df.groupby('label')[['nb_caracteres', 'nb_mots']].agg(['mean', 'median']))

# Questions de réflexion (réponses à compléter / déjà indiquées dans l'énoncé)
# Q: Quel est le pourcentage de spam ? → ~13.4%
# Q: Pourquoi ce déséquilibre pose problème ? → Le modèle peut tout prédire "ham" et avoir une bonne accuracy.
# Q: Les messages spam sont-ils plus longs ou plus courts ? → En général plus longs (à vérifier sur les données).

# 1.3 Visualisations

# 1. Distribution de la variable cible (bar plot)
plt.figure(figsize=(8, 5))
df['label'].value_counts().plot(kind='bar', color=['steelblue', 'coral'], edgecolor='black')
plt.title('Distribution de la variable cible (spam vs ham)', fontsize=14, fontweight='bold')
plt.xlabel('Label')
plt.ylabel('Nombre de messages')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Histogramme nombre de caractères (spam vs ham)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df[df['label'] == 'ham']['nb_caracteres'].hist(bins=50, alpha=0.7, label='Ham', color='steelblue', density=True)
df[df['label'] == 'spam']['nb_caracteres'].hist(bins=50, alpha=0.7, label='Spam', color='coral', density=True)
plt.xlabel('Nombre de caractères')
plt.ylabel('Densité')
plt.title('Longueur des messages (caractères)')
plt.legend()
plt.subplot(1, 2, 2)
df[df['label'] == 'ham']['nb_mots'].hist(bins=50, alpha=0.7, label='Ham', color='steelblue', density=True)
df[df['label'] == 'spam']['nb_mots'].hist(bins=50, alpha=0.7, label='Spam', color='coral', density=True)
plt.xlabel('Nombre de mots')
plt.ylabel('Densité')
plt.title('Longueur des messages (mots)')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Top 15-20 mots les plus fréquents (spam et ham)
def get_word_frequency(messages, top_n=20):
    text = ' '.join(messages.astype(str))
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

spam_messages = df[df['label'] == 'spam']['message']
ham_messages = df[df['label'] == 'ham']['message']
top_spam_words = get_word_frequency(spam_messages, top_n=20)
top_ham_words = get_word_frequency(ham_messages, top_n=20)

print("\nTop 20 mots les plus fréquents - SPAM:")
for i, (word, count) in enumerate(top_spam_words, 1):
    print(f"  {i}. {word}: {count}")

print("\nTop 20 mots les plus fréquents - HAM:")
for i, (word, count) in enumerate(top_ham_words, 1):
    print(f"  {i}. {word}: {count}")

# Graphique top mots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
words_spam = [w[0] for w in top_spam_words]
counts_spam = [w[1] for w in top_spam_words]
axes[0].barh(range(len(words_spam)), counts_spam, color='coral', alpha=0.8)
axes[0].set_yticks(range(len(words_spam)))
axes[0].set_yticklabels(words_spam)
axes[0].invert_yaxis()
axes[0].set_title('Top 20 mots - Spam')
axes[0].set_xlabel('Fréquence')

words_ham = [w[0] for w in top_ham_words]
counts_ham = [w[1] for w in top_ham_words]
axes[1].barh(range(len(words_ham)), counts_ham, color='steelblue', alpha=0.8)
axes[1].set_yticks(range(len(words_ham)))
axes[1].set_yticklabels(words_ham)
axes[1].invert_yaxis()
axes[1].set_title('Top 20 mots - Ham')
axes[1].set_xlabel('Fréquence')
plt.tight_layout()
plt.show()

# 1.4 Préparation des données

# Encoder la cible : spam=1, ham=0
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Vectorisation TF-IDF (fit sur tout le corpus comme dans l'énoncé, puis split)
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df['message'])

print(f"\nShape de X: {X.shape}")
print(f"Type de X: {type(X)}")

# Split train/test stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribution train: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Distribution test: {pd.Series(y_test).value_counts().to_dict()}")

print("\n✅ PARTIE 1 terminée")

# =============================================================================
# PARTIE 2 : ALGORITHMES AVANCÉS
# =============================================================================

print("\n" + "="*80)
print("PARTIE 2 : ALGORITHMES AVANCÉS")
print("="*80)

# 2.1 SVM - on teste les 3 noyaux comme demandé
# SVM Linear
svm_linear = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
t0 = time()
svm_linear.fit(X_train, y_train)
print(f"SVM Linear entraîné en {time()-t0:.2f}s")
y_pred_linear = svm_linear.predict(X_test)
y_proba_linear = svm_linear.predict_proba(X_test)[:, 1]
print(f"  F1={f1_score(y_test, y_pred_linear):.4f}, AUC={roc_auc_score(y_test, y_proba_linear):.4f}")

# SVM RBF (configuration de base pour les graphiques)
svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
t0 = time()
svm_model.fit(X_train, y_train)
svm_time = time() - t0
print(f"SVM RBF entraîné en {svm_time:.2f}s")

y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_proba_svm)

print(f"Accuracy: {svm_accuracy:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}")
print(f"F1-Score: {svm_f1:.4f}, ROC-AUC: {svm_auc:.4f}")

plot_confusion_matrix(y_test, y_pred_svm, title="SVM RBF - Matrice de Confusion")
plot_roc_curve(y_test, y_proba_svm, model_name="SVM RBF")

# SVM Polynomial
svm_poly = SVC(kernel='poly', degree=3, C=1.0, class_weight='balanced', probability=True, random_state=RANDOM_STATE)
t0 = time()
svm_poly.fit(X_train, y_train)
print(f"SVM Poly entraîné en {time()-t0:.2f}s")
y_pred_poly = svm_poly.predict(X_test)
y_proba_poly = svm_poly.predict_proba(X_test)[:, 1]
print(f"  F1={f1_score(y_test, y_pred_poly):.4f}, AUC={roc_auc_score(y_test, y_proba_poly):.4f}")

# Question 2.1 : Quel noyau performe le mieux ? → Linear souvent meilleur pour le texte (données haute dimension linéairement séparables).

# 2.2 KNN
k_values = [3, 5, 10, 20, 50]
knn_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_scores.append(f1_score(y_test, y_pred))
    print(f"  K={k}: F1={knn_scores[-1]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel('Nombre de voisins (K)')
plt.ylabel('F1-Score')
plt.title('Performance KNN en fonction de K')
plt.grid(alpha=0.3)
plt.show()

best_k = k_values[np.argmax(knn_scores)]
print(f"\nMeilleur K: {best_k}")

knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_proba_knn = knn_model.predict_proba(X_test)[:, 1]
knn_f1 = f1_score(y_test, y_pred_knn)
knn_auc = roc_auc_score(y_test, y_proba_knn)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
print(f"KNN (K={best_k}) F1={knn_f1:.4f}, AUC={knn_auc:.4f}")

# 2.3 XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)
t0 = time()
xgb_model.fit(X_train, y_train)
xgb_time = time() - t0
print(f"XGBoost entraîné en {xgb_time:.2f}s")

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
print(f"F1-Score: {xgb_f1:.4f}, ROC-AUC: {xgb_auc:.4f}")

# Top 20 mots importants XGBoost
feature_names = vectorizer.get_feature_names_out()
importances = xgb_model.feature_importances_
top_indices = np.argsort(importances)[::-1][:20]
print("\nTop 20 mots les plus importants (XGBoost):")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

# 2.4 LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbose=-1
)
t0 = time()
lgbm_model.fit(X_train, y_train)
lgbm_time = time() - t0
y_pred_lgbm = lgbm_model.predict(X_test)
y_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_f1 = f1_score(y_test, y_pred_lgbm)
lgbm_auc = roc_auc_score(y_test, y_proba_lgbm)
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
lgbm_precision = precision_score(y_test, y_pred_lgbm)
lgbm_recall = recall_score(y_test, y_pred_lgbm)
print(f"LightGBM entraîné en {lgbm_time:.2f}s")
print(f"F1-Score: {lgbm_f1:.4f}, ROC-AUC: {lgbm_auc:.4f}")
print(f"Comparaison vitesse: XGBoost={xgb_time:.2f}s vs LightGBM={lgbm_time:.2f}s")

# 2.5 Comparaison globale
results = {
    'SVM': {
        'accuracy': svm_accuracy,
        'precision': svm_precision,
        'recall': svm_recall,
        'f1': svm_f1,
        'auc': svm_auc
    },
    'KNN': {
        'accuracy': knn_accuracy,
        'precision': knn_precision,
        'recall': knn_recall,
        'f1': knn_f1,
        'auc': knn_auc
    },
    'XGBoost': {
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1': xgb_f1,
        'auc': xgb_auc
    },
    'LightGBM': {
        'accuracy': lgbm_accuracy,
        'precision': lgbm_precision,
        'recall': lgbm_recall,
        'f1': lgbm_f1,
        'auc': lgbm_auc
    }
}
compare_models_performance(results)

print("\n✅ PARTIE 2 terminée")

# =============================================================================
# PARTIE 3 : OPTIMISATION & DIMENSIONNALITÉ
# =============================================================================

print("\n" + "="*80)
print("PARTIE 3 : OPTIMISATION & DIMENSIONNALITÉ")
print("="*80)

# 3.1 RandomizedSearchCV sur XGBoost
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

t0 = time()
random_search.fit(X_train, y_train)
optim_time = time() - t0
print(f"\nOptimisation terminée en {optim_time:.2f}s")

print("\nMeilleurs hyperparamètres:")
for param, value in random_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"Meilleur score CV (F1): {random_search.best_score_:.4f}")
print(f"Score de base: {xgb_f1:.4f}")

# Évaluation sur le test set
xgb_optimized = random_search.best_estimator_
y_pred_xgb_opt = xgb_optimized.predict(X_test)
y_proba_xgb_opt = xgb_optimized.predict_proba(X_test)[:, 1]
xgb_opt_f1 = f1_score(y_test, y_pred_xgb_opt)
xgb_opt_auc = roc_auc_score(y_test, y_proba_xgb_opt)
print(f"\nTest set - F1 optimisé: {xgb_opt_f1:.4f}, ROC-AUC: {xgb_opt_auc:.4f}")
plot_confusion_matrix(y_test, y_pred_xgb_opt, title="XGBoost Optimisé")

# 3.2 TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)
variance_explained = svd.explained_variance_ratio_.sum()
print(f"\nDimensions originales: {X_train.shape[1]}, après SVD: {X_train_svd.shape[1]}")
print(f"Variance expliquée: {variance_explained:.4f} ({variance_explained*100:.2f}%)")

cumsum_variance = np.cumsum(svd.explained_variance_ratio_)
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumsum_variance)+1), cumsum_variance, marker='o', linewidth=2)
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('Variance Expliquée par TruncatedSVD')
plt.grid(alpha=0.3)
plt.show()

xgb_svd = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)
t0 = time()
xgb_svd.fit(X_train_svd, y_train)
svd_time = time() - t0
y_pred_svd = xgb_svd.predict(X_test_svd)
svd_f1 = f1_score(y_test, y_pred_svd)
print(f"Sans SVD: F1={xgb_f1:.4f}, Temps={xgb_time:.2f}s | Avec SVD: F1={svd_f1:.4f}, Temps={svd_time:.2f}s")

# 3.3 Feature selection
selector = SelectKBest(score_func=chi2, k=500)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_mask = selector.get_support()
selected_features = [feature_names[i] for i, sel in enumerate(selected_mask) if sel]
print(f"\nFeature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")
print(f"Exemples de mots sélectionnés: {selected_features[:20]}")

xgb_fs = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)
xgb_fs.fit(X_train_selected, y_train)
y_pred_fs = xgb_fs.predict(X_test_selected)
fs_f1 = f1_score(y_test, y_pred_fs)
print(f"Toutes features: F1={xgb_f1:.4f} | 500 features: F1={fs_f1:.4f}")

print("\n✅ PARTIE 3 terminée")

# =============================================================================
# PARTIE 4 : INTRODUCTION MLOps
# =============================================================================

print("\n" + "="*80)
print("PARTIE 4 : INTRODUCTION MLOps")
print("="*80)

# 4.1 MLflow Tracking
import mlflow
import mlflow.sklearn

mlflow.set_experiment("spam_detection_tp5")

with mlflow.start_run(run_name="xgboost_baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("f1_score", xgb_f1)
    mlflow.log_metric("roc_auc", xgb_auc)
    mlflow.sklearn.log_model(xgb_model, "model")
    print("✓ Run xgboost_baseline logged")

with mlflow.start_run(run_name="xgboost_optimized"):
    for param, value in random_search.best_params_.items():
        mlflow.log_param(param, value)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("optimization_method", "RandomizedSearchCV")
    mlflow.log_metric("f1_score", xgb_opt_f1)
    mlflow.log_metric("roc_auc", xgb_opt_auc)
    mlflow.log_metric("f1_improvement", xgb_opt_f1 - xgb_f1)
    mlflow.sklearn.log_model(xgb_optimized, "model")
    print("✓ Run xgboost_optimized logged")

# 4.2 Pipeline de production
production_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ('model', xgb_optimized)
])

messages = df['message'].values
labels = le.transform(df['label'])
messages_train, messages_test, y_train_raw, y_test_raw = train_test_split(
    messages, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
)

print("Entraînement du pipeline sur données brutes...")
production_pipeline.fit(messages_train, y_train_raw)
y_pred_pipeline = production_pipeline.predict(messages_test)
pipeline_f1 = f1_score(y_test_raw, y_pred_pipeline)
print(f"Pipeline F1-Score: {pipeline_f1:.4f}")

# Test avec de nouveaux messages
new_messages = [
    "Congratulations! You've won a FREE prize! Call now!",
    "Hey, are we still meeting for lunch tomorrow?"
]
predictions = production_pipeline.predict(new_messages)
for msg, pred in zip(new_messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}] {msg}")

joblib.dump(production_pipeline, 'spam_detector_pipeline.pkl')
print("✓ Pipeline sauvegardé: spam_detector_pipeline.pkl")
loaded_pipeline = joblib.load('spam_detector_pipeline.pkl')
print("✓ Pipeline rechargé OK")

# 4.3 Monitoring - Data Drift (longueur des messages train vs test)
train_lengths = np.array([len(m) for m in messages_train])
test_lengths = np.array([len(m) for m in messages_test])

print("Longueur moyenne des messages:")
print(f"Train: {train_lengths.mean():.2f}")
print(f"Test: {test_lengths.mean():.2f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(train_lengths, bins=30, alpha=0.6, label='Train', density=True)
plt.hist(test_lengths, bins=30, alpha=0.6, label='Test', density=True)
plt.xlabel('Longueur du message')
plt.ylabel('Densité')
plt.title('Distribution de la longueur des messages')
plt.legend()
plt.subplot(1, 2, 2)
plt.boxplot([train_lengths, test_lengths], labels=['Train', 'Test'])
plt.ylabel('Longueur')
plt.title('Boxplot longueur messages')
plt.tight_layout()
plt.show()

# Simuler un drift avec de nouveaux types de spam
new_spam_messages = [
    "URGENT: Buy Bitcoin now! 1000% guaranteed returns! Limited time!",
    "COVID-19 vaccine available NOW! Click here for instant access!",
    "Get rich quick with NFTs! Join our exclusive group!",
]
predictions_drift = production_pipeline.predict(new_spam_messages)
probas_drift = production_pipeline.predict_proba(new_spam_messages)
for msg, pred, proba in zip(new_spam_messages, predictions_drift, probas_drift):
    label = "SPAM" if pred == 1 else "HAM"
    conf = proba[pred]
    print(f"[{label}] (confiance: {conf:.2f}) {msg}")

# 4.4 Versioning et reproductibilité
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

print("\n✅ PARTIE 4 terminée")
print("\n" + "="*80)
print("TP5 TERMINÉ - Livrables: script, spam_detector_pipeline.pkl, runs MLflow, README")
print("="*80)
