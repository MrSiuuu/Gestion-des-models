# TP4 - Détection de fraude bancaire

Projet de Machine Learning pour la détection de transactions frauduleuses sur des données bancaires anonymisées. Réalisé dans le cadre du Master 1 Data Engineering (Concepts et technologies IA).

## Objectifs

- Construire un modèle qui détecte au moins 85 % des fraudes (Recall >= 0.85).
- Limiter les faux positifs (Precision la plus élevée possible).
- Utiliser du feature engineering, gérer le déséquilibre des classes et optimiser les hyperparamètres.

## Structure du projet

```
Jour4/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── api_fraud_detection.py    # API Flask pour prédire en production
├── SUJET_TP4_DETECTION_FRAUDE.md   # Sujet détaillé du TP
├── AIDE_MEMOIRE.md           # Rappels techniques (syntaxe, formules)
│
├── data/
│   ├── README_DATA.md        # Description du jeu de données
│   ├── download_data.py      # Téléchargement du CSV depuis Kaggle
│   └── creditcard.csv        # À télécharger (absent du dépôt, > 100 Mo)
│
├── notebooks/
│   ├── TP4_Detection_Fraude_ETUDIANT.ipynb   # Notebook principal (EDA, modèles, optimisation)
│   └── README_NOTEBOOK.md    # Instructions pour le notebook
│
├── utils/
│   ├── feature_engineering.py # Création des features (temporel, montants, interactions)
│   ├── predict.py            # Classe FraudDetector (chargement modèle, prédiction)
│   └── __init__.py
│
├── tests/
│   └── test_fraud_detector.py # Tests unitaires du détecteur
│
└── models/                   # Modèles sauvegardés (créé après entraînement)
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/MrSiuuu/Gestion-des-models.git
cd Gestion-des-models
```

### 2. Environnement virtuel et dépendances

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(Sur Linux/Mac : `source venv/bin/activate`.)

### 3. Données

Le fichier `data/creditcard.csv` (environ 144 Mo) n’est pas présent sur GitHub (limite de taille). Deux possibilités :

- **Automatique** : aller dans `data/` et lancer `python download_data.py` (nécessite un compte Kaggle et un token API).
- **Manuel** : télécharger [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) sur Kaggle, puis placer `creditcard.csv` dans le dossier `data/`.

Voir `data/README_DATA.md` pour le détail.

## Utilisation

### Lancer le notebook

```bash
jupyter notebook notebooks/TP4_Detection_Fraude_ETUDIANT.ipynb
```

Le notebook couvre : exploration, feature engineering, comparaison de 6 modèles (dont Random Forest et XGBoost), optimisation (GridSearch, RandomizedSearch), importance des variables (MDI, permutation, SHAP), courbes d’apprentissage, calibration, réglage du seuil, sauvegarde du modèle et validation temporelle.

### Utiliser le modèle entraîné (API)

Après avoir exécuté le notebook et sauvegardé le modèle dans `models/` :

```bash
python api_fraud_detection.py
```

L’API écoute sur `http://localhost:5000`. Voir le code pour les endpoints (ex. prédiction par JSON).

### Lancer les tests

```bash
pytest tests/ -v
```

## Données (résumé)

- **Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **Volume** : 284 807 transactions, 492 fraudes (environ 0,17 %).
- **Variables** : `Time`, `V1` à `V28` (PCA), `Amount`, `Class` (0 = légitime, 1 = fraude).

Le déséquilibre est fort ; le TP utilise notamment `class_weight`, SMOTE et la métrique PR-AUC.

## Technologies

- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, xgboost
- Flask (API), joblib (sauvegarde des modèles), pytest (tests)

## Auteur

KOUYATE Issa – Master 1 Data Engineering.

## Licence

Projet pédagogique. Voir le sujet du TP pour le cadre d’utilisation.
