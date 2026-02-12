# TP4 - Détection de fraude bancaire

**Master 1 Data Engineering - YNOV Montpellier**  
**Sujet : Détection de fraude sur transactions bancaires**

---

## Description

Ce projet réalise un système de détection de fraude à partir de transactions bancaires anonymisées. Il met en œuvre :

- Analyse exploratoire et feature engineering (features temporelles, montants, interactions, agrégations)
- Gestion du déséquilibre des classes (class_weight, SMOTE) et métriques adaptées (PR-AUC)
- Comparaison de 6 algorithmes : régression logistique, arbre de décision, Random Forest, SVM, KNN, XGBoost
- Optimisation des hyperparamètres (GridSearchCV, RandomizedSearchCV) et pipelines scikit-learn
- Explicabilité : importance des variables (MDI, permutation, SHAP), courbes d’apprentissage, calibration
- Déploiement : sérialisation du modèle (joblib), API Flask, tests unitaires

Le jeu de données utilisé est le **Credit Card Fraud Detection** (Kaggle) : 284 807 transactions, 492 fraudes (environ 0,17 %), avec des variables PCA (V1–V28), Time et Amount.

---

## Prérequis

- Python 3.8 ou supérieur
- pip à jour

Un environnement virtuel est recommandé mais pas obligatoire.

---

## Installation

### 1. Cloner ou récupérer le projet

Placer tous les fichiers du projet dans un même dossier (par exemple en clonant le dépôt puis en ouvrant le dossier **Jour4**).

### 2. Installer les dépendances

Ouvrir un terminal dans le dossier **Jour4** et exécuter :

```bash
pip install -r requirements.txt
```

En cas d’erreur liée à setuptools (par exemple sous Python 3.14), exécuter d’abord :

```bash
pip install --upgrade setuptools wheel
```

puis à nouveau :

```bash
pip install -r requirements.txt
```

### 3. Vérifier l’installation

```bash
python -c "import sklearn, pandas, imblearn, xgboost, flask; print('OK')"
```

Si aucun message d’erreur n’apparaît, l’environnement est prêt.

---

## Dataset

Le notebook et l’API attendent le fichier **creditcard.csv** dans le dossier **data/**.

- **Source :** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Téléchargement manuel :** se connecter à Kaggle, télécharger le dataset, puis placer le fichier **creditcard.csv** dans le dossier **data/** du projet.
- **Script automatique :** exécuter `python download_data.py` depuis le dossier **data/** (nécessite un compte Kaggle et un token API configuré).

Le fichier fait environ 144 Mo et n’est pas fourni dans le dépôt (limite GitHub 100 Mo). Sans ce fichier, le notebook et l’API ne pourront pas charger les données.

Voir **data/README_DATA.md** pour la structure des colonnes et les statistiques.

---

## Exécution du notebook principal

### Lancer le TP (Parties 1 à 5)

Dans un terminal, se placer dans le dossier **Jour4** puis exécuter :

```bash
jupyter notebook notebooks/TP4_Detection_Fraude_ETUDIANT.ipynb
```

- **Partie 1 :** chargement des données, analyse du déséquilibre, corrélations, feature engineering (15+ features).
- **Partie 2 :** préparation train/test, scaling (RobustScaler), entraînement et évaluation de 6 modèles (LR, DT, RF, SVM, KNN, XGBoost), comparaison SMOTE vs class_weight.
- **Partie 3 :** pipeline ML, GridSearchCV et RandomizedSearchCV, optimisation XGBoost, validation (StratifiedKFold, TimeSeriesSplit).
- **Partie 4 :** importance des variables (MDI, permutation, SHAP), courbes ROC et Precision-Recall, learning curves, calibration, optimisation du seuil de décision.
- **Partie 5 :** validation temporelle, sérialisation du modèle (joblib), métadonnées.

Répondre aux questions dans les cellules markdown prévues à cet effet. À la fin de la Partie 5, le modèle est sauvegardé dans **models/fraud_detector_v1.joblib**.

### Lancer l’API (après entraînement)

Une fois le modèle sauvegardé dans **models/** :

```bash
python api_fraud_detection.py
```

L’API écoute sur **http://localhost:5000**. Consulter le code pour les endpoints (prédiction par JSON).

### Lancer les tests

```bash
pytest tests/ -v
```

---

## Structure du projet

```
Jour4/
|
|-- README.md                         (ce fichier)
|-- requirements.txt                  (dépendances Python)
|-- api_fraud_detection.py            (API Flask pour prédiction)
|-- SUJET_TP4_DETECTION_FRAUDE.md     (sujet détaillé du TP)
|-- AIDE_MEMOIRE.md                   (rappels techniques)
|
|-- data/
|   |-- README_DATA.md                (description du dataset)
|   |-- download_data.py              (téléchargement Kaggle)
|   |-- creditcard.csv                 (à télécharger, non fourni dans le dépôt)
|
|-- notebooks/
|   |-- TP4_Detection_Fraude_ETUDIANT.ipynb   (notebook principal)
|   |-- README_NOTEBOOK.md             (instructions notebook)
|
|-- utils/
|   |-- feature_engineering.py        (création des features)
|   |-- predict.py                    (classe FraudDetector)
|   |-- __init__.py
|
|-- tests/
|   |-- test_fraud_detector.py         (tests unitaires)
|
|-- models/                            (créé après entraînement)
|   |-- fraud_detector_v1.joblib      (généré après exécution du notebook)
```

---

## Livrables du TP

- **TP4_Detection_Fraude_ETUDIANT.ipynb** : notebook complété (code + réponses aux questions).
- **models/fraud_detector_v1.joblib** : pipeline de détection de fraude (scaler + modèle), généré en fin de Partie 5.
- **README** : ce fichier, décrivant l’installation et l’exécution du projet.

---

## Auteur

KOUYATE Issa – Master 1 Data Engineering.
