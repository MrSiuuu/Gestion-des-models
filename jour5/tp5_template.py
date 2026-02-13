"""
TP5 - Techniques Avancées & MLOps
Master 1 Data Engineering - YNOV Montpellier

Nom & Prénom : ___________________
Date : ___________________

Objectif : Développer un système de détection de fraude bancaire
"""

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from time import time

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Imports Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Imports des algorithmes
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Imports optimisation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Imports pipeline
from sklearn.pipeline import Pipeline

# Imports utilitaires locaux
from utils import (
    load_fraud_dataset,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models_performance,
    plot_feature_importance,
    detect_data_drift,
    save_model_info
)

print("✓ Tous les imports réussis")

# =============================================================================
# PARTIE 1 : EXPLORATION ET PRÉPARATION (45 min)
# =============================================================================

print("\n" + "="*80)
print("PARTIE 1 : EXPLORATION ET PRÉPARATION")
print("="*80)

# 1.1 Chargement des données
# TODO: Charger le dataset avec load_fraud_dataset()
# Pour tests rapides: sample_size=50000
df = None  # À COMPLÉTER

# TODO: Afficher les premières lignes


# TODO: Afficher les informations générales (shape, dtypes, describe, valeurs manquantes)


# 1.2 Analyse du déséquilibre
# TODO: Analyser la distribution de la classe cible
# Compter les occurrences, calculer les pourcentages et le ratio


# QUESTION 1.1: Quel est le pourcentage de transactions frauduleuses ?
# RÉPONSE:

# QUESTION 1.2: Pourquoi ce déséquilibre pose-t-il problème ?
# RÉPONSE:

# 1.3 Visualisations
# TODO: Créer 3-4 visualisations pertinentes
# - Distribution de la classe cible
# - Distribution des montants par classe
# - Distribution temporelle des fraudes
# - Matrice de corrélation (échantillon)


# 1.4 Préparation des données
# TODO: Séparer features (X) et cible (y)
X = None  # À COMPLÉTER
y = None  # À COMPLÉTER

# TODO: Split train/test stratifié (80/20)
X_train, X_test, y_train, y_test = None, None, None, None  # À COMPLÉTER

# TODO: Normaliser UNIQUEMENT Time et Amount
# Les features V1-V28 sont déjà normalisées !


print("\n✅ CHECKPOINT PARTIE 1 : Données préparées")

# =============================================================================
# PARTIE 2 : ALGORITHMES AVANCÉS (90 min)
# =============================================================================

print("\n" + "="*80)
print("PARTIE 2 : ALGORITHMES AVANCÉS")
print("="*80)

# Dictionnaire pour stocker les résultats
results = {}

# 2.1 Support Vector Machine (SVM)
print("\n--- 2.1 SVM ---")

# TODO: Créer et entraîner un SVM
# Paramètres: kernel='rbf', C=1.0, class_weight='balanced', probability=True
start_time = time()
svm_model = None  # À COMPLÉTER
# svm_model.fit(X_train, y_train)
svm_time = time() - start_time

# TODO: Évaluer le SVM
y_pred_svm = None  # À COMPLÉTER
y_proba_svm = None  # À COMPLÉTER ([:,1] pour la classe positive)

svm_accuracy = None  # À COMPLÉTER
svm_precision = None  # À COMPLÉTER
svm_recall = None  # À COMPLÉTER
svm_f1 = None  # À COMPLÉTER
svm_auc = None  # À COMPLÉTER

print(f"SVM - F1: {svm_f1:.4f}, AUC: {svm_auc:.4f}, Temps: {svm_time:.2f}s")

# TODO: Afficher la matrice de confusion
# plot_confusion_matrix(y_test, y_pred_svm, title="SVM")

# Stocker les résultats
results['SVM'] = {
    'accuracy': svm_accuracy,
    'precision': svm_precision,
    'recall': svm_recall,
    'f1': svm_f1,
    'auc': svm_auc
}

# QUESTION 2.1: Quel noyau SVM performe le mieux ? (Testez linear, rbf, poly)
# RÉPONSE:

# 2.2 K-Nearest Neighbors (KNN)
print("\n--- 2.2 KNN ---")

# TODO: Tester différentes valeurs de K
k_values = [3, 5, 10, 20, 50]
knn_scores = []

for k in k_values:
    # TODO: Créer, entraîner et évaluer KNN
    # Paramètres: n_neighbors=k, weights='distance'
    pass

# TODO: Identifier le meilleur K
best_k = None  # À déterminer

# TODO: Évaluer complètement le meilleur KNN
knn_model = None  # À COMPLÉTER
y_pred_knn = None
y_proba_knn = None

knn_f1 = None  # À COMPLÉTER
knn_auc = None  # À COMPLÉTER

print(f"KNN (k={best_k}) - F1: {knn_f1:.4f}, AUC: {knn_auc:.4f}")

# QUESTION 2.2: Pourquoi un K trop petit ou trop grand est problématique ?
# RÉPONSE:

# QUESTION 2.3: KNN est-il adapté pour ce problème ? Justifiez.
# RÉPONSE:

# 2.3 XGBoost
print("\n--- 2.3 XGBoost ---")

# TODO: Calculer scale_pos_weight
scale_pos_weight = None  # n_negative / n_positive

# TODO: Créer et entraîner XGBoost
start_time = time()
xgb_model = None  # À COMPLÉTER
# Paramètres: n_estimators=100, max_depth=5, learning_rate=0.1,
#            scale_pos_weight=..., random_state=42
xgb_time = time() - start_time

# TODO: Évaluer XGBoost
y_pred_xgb = None
y_proba_xgb = None

xgb_f1 = None
xgb_auc = None

print(f"XGBoost - F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}, Temps: {xgb_time:.2f}s")

# TODO: Afficher l'importance des features (top 10)
# feature_names = X.columns.tolist()
# importances = xgb_model.feature_importances_
# plot_feature_importance(feature_names, importances, top_n=10)

# 2.4 LightGBM
print("\n--- 2.4 LightGBM ---")

# TODO: Créer et entraîner LightGBM
start_time = time()
lgbm_model = None  # À COMPLÉTER
# Paramètres: n_estimators=100, max_depth=5, learning_rate=0.1,
#            scale_pos_weight=..., random_state=42, verbose=-1
lgbm_time = time() - start_time

# TODO: Évaluer LightGBM
y_pred_lgbm = None
y_proba_lgbm = None

lgbm_f1 = None
lgbm_auc = None

print(f"LightGBM - F1: {lgbm_f1:.4f}, AUC: {lgbm_auc:.4f}, Temps: {lgbm_time:.2f}s")
print(f"Comparaison vitesse: XGBoost={xgb_time:.2f}s vs LightGBM={lgbm_time:.2f}s")

# 2.5 Comparaison globale
print("\n--- 2.5 Comparaison Globale ---")

# TODO: Compléter le dictionnaire results avec tous les modèles
results['KNN'] = {'f1': knn_f1, 'auc': knn_auc}  # Compléter avec toutes les métriques
results['XGBoost'] = {'f1': xgb_f1, 'auc': xgb_auc}  # Idem
results['LightGBM'] = {'f1': lgbm_f1, 'auc': lgbm_auc}  # Idem

# TODO: Afficher la comparaison
# compare_models_performance(results)

# QUESTION 2.4: Quel modèle performe le mieux ? Selon quelle métrique ?
# RÉPONSE:

# QUESTION 2.5: Pourquoi l'Accuracy n'est pas une bonne métrique ici ?
# RÉPONSE:

# QUESTION 2.6: Trade-off Precision vs Recall dans la détection de fraude ?
# RÉPONSE:

print("\n✅ CHECKPOINT PARTIE 2 : 4 algorithmes évalués")

# =============================================================================
# PARTIE 3 : OPTIMISATION & DIMENSIONNALITÉ (60 min)
# =============================================================================

print("\n" + "="*80)
print("PARTIE 3 : OPTIMISATION & DIMENSIONNALITÉ")
print("="*80)

# 3.1 RandomizedSearchCV sur XGBoost
print("\n--- 3.1 RandomizedSearchCV ---")

# TODO: Définir l'espace de recherche
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# TODO: Lancer RandomizedSearchCV
# Paramètres: n_iter=50, cv=5, scoring='f1', n_jobs=-1, verbose=1
print("Lancement de RandomizedSearchCV (plusieurs minutes)...")

xgb_base = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)
random_search = None  # À COMPLÉTER (RandomizedSearchCV)

# random_search.fit(X_train, y_train)

# TODO: Afficher les meilleurs hyperparamètres
# print("Meilleurs hyperparamètres:", random_search.best_params_)
# print(f"Meilleur score CV: {random_search.best_score_:.4f}")

# TODO: Évaluer le modèle optimisé
# xgb_optimized = random_search.best_estimator_
# y_pred_opt = xgb_optimized.predict(X_test)
# xgb_opt_f1 = f1_score(y_test, y_pred_opt)
# print(f"F1-Score optimisé: {xgb_opt_f1:.4f}")

# QUESTION 3.1: Pourquoi RandomizedSearchCV plutôt que GridSearchCV ?
# RÉPONSE:

# 3.2 Réduction de Dimension avec PCA
print("\n--- 3.2 PCA ---")

# TODO: Appliquer PCA pour conserver 95% de variance
pca = None  # À COMPLÉTER (PCA avec n_components=0.95)

# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# print(f"Dimensions originales: {X_train.shape[1]}")
# print(f"Dimensions après PCA: {X_train_pca.shape[1]}")
# print(f"Variance expliquée: {pca.explained_variance_ratio_.sum():.4f}")

# TODO: Entraîner XGBoost sur données PCA et comparer


# QUESTION 3.2: Dans quel contexte PCA est recommandé avant KNN ?
# RÉPONSE:

# 3.3 Feature Selection
print("\n--- 3.3 Feature Selection ---")

# TODO: Sélectionner les K=15 meilleures features
selector = None  # À COMPLÉTER (SelectKBest, k=15)

# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)

# TODO: Identifier et afficher les features sélectionnées


# TODO: Entraîner un modèle sur les features sélectionnées et comparer


print("\n✅ CHECKPOINT PARTIE 3 : Optimisation effectuée")

# =============================================================================
# PARTIE 4 : INTRODUCTION MLOPS (45 min)
# =============================================================================

print("\n" + "="*80)
print("PARTIE 4 : INTRODUCTION MLOPS")
print("="*80)

# 4.1 MLflow Tracking
print("\n--- 4.1 MLflow Tracking ---")

import mlflow
import mlflow.sklearn

# TODO: Configurer MLflow
mlflow.set_experiment("fraud_detection_tp5")

# TODO: Tracker une expérimentation
# with mlflow.start_run(run_name="xgboost_baseline"):
#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_param("max_depth", 5)
#     mlflow.log_metric("f1_score", xgb_f1)
#     mlflow.log_metric("roc_auc", xgb_auc)
#     mlflow.sklearn.log_model(xgb_model, "model")

print("✓ Pour voir l'UI MLflow: mlflow ui (puis http://localhost:5000)")

# 4.2 Pipeline de Production
print("\n--- 4.2 Pipeline de Production ---")

# TODO: Créer un pipeline complet Preprocessing + Model


# TODO: Sauvegarder le pipeline
import joblib
# joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')

# 4.3 Monitoring - Data Drift
print("\n--- 4.3 Data Drift ---")

# TODO: Simuler un drift en modifiant certaines features
# X_production = X_test.copy()
# X_production['Amount'] = X_production['Amount'] * 1.5

# TODO: Détecter et visualiser le drift
# detect_data_drift(X_train, X_production, feature='Amount')

# QUESTION 4.1: Comment détecter automatiquement un drift en production ?
# RÉPONSE:

# 4.4 Versioning et Reproductibilité
print("\n--- 4.4 Versioning ---")

# TODO: Sauvegarder les métadonnées du modèle
# metadata = {
#     'f1_score': xgb_f1,
#     'roc_auc': xgb_auc,
#     'training_samples': len(X_train)
# }
# save_model_info(xgb_model, 'model_info.json', metadata=metadata)

print("\n✅ CHECKPOINT PARTIE 4 : MLOps appliqué")

# =============================================================================
# SYNTHÈSE FINALE
# =============================================================================

print("\n" + "="*80)
print("SYNTHÈSE FINALE")
print("="*80)

print("""
Résumé de vos résultats :

Meilleur modèle : _________________

Performances :
- F1-Score : _______
- ROC-AUC : _______
- Precision : _______
- Recall : _______

Optimisations appliquées :
[ ] RandomizedSearchCV
[ ] PCA
[ ] Feature Selection
[ ] Gestion déséquilibre

MLOps :
[ ] MLflow tracking
[ ] Pipeline production
[ ] Monitoring drift
[ ] Versioning modèle

Difficultés rencontrées :
1. 
2. 
3. 

Pistes d'amélioration :
1. 
2. 
3. 
""")

print("\n🎉 Félicitations ! TP5 terminé !")
print("N'oubliez pas de sauvegarder vos résultats et soumettre les livrables.")
