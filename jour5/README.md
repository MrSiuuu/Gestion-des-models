# TP5 - Techniques Avancées et MLOps

**Master 1 Data Engineering - YNOV Montpellier**  
**Enseignant : BENHAMDI Ayoub**  
**Sujet : Détection de Spam SMS**

---

## Description

Ce projet réalise un système de détection de spam SMS à partir du contenu textuel des messages. Il met en œuvre :

- Traitement de données textuelles (vectorisation TF-IDF)
- Algorithmes de classification : SVM, KNN, XGBoost, LightGBM
- Optimisation des hyperparamètres (RandomizedSearchCV)
- Réduction de dimensionnalité (TruncatedSVD) et sélection de variables
- MLOps : suivi des expériences (MLflow), pipeline de production, versioning

Le jeu de données utilisé est le **SMS Spam Collection Dataset** (Kaggle) : environ 5 572 messages en anglais, étiquetés spam ou ham, avec un déséquilibre de classes (environ 13,4 % de spam).

---

## Prérequis

- Python 3.8 ou supérieur
- pip à jour

Un environnement virtuel est recommandé mais pas obligatoire.

---

## Installation

### 1. Cloner ou récupérer le projet

Placer tous les fichiers du projet dans un même dossier (par exemple `Jour5/Etudiants`).

### 2. Installer les dépendances

Ouvrir un terminal dans le dossier du projet et exécuter :

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
python -c "import sklearn, xgboost, lightgbm, mlflow, pandas; print('OK')"
```

Si aucun message d’erreur n’apparaît, l’environnement est prêt.

---

## Dataset

Le script attend le fichier **spam.csv** dans le dossier **data/**.

- **Source :** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Téléchargement manuel :** se connecter à Kaggle, télécharger le dataset, puis placer le fichier **spam.csv** dans le dossier **data/** du projet.
- **Structure attendue :** le CSV doit contenir au minimum une colonne de type (spam/ham) et une colonne de message (texte). La fonction `load_spam_dataset` dans `utils.py` gère les noms de colonnes courants (v1/v2 ou label/message).

Sans ce fichier, le script s’arrêtera avec une erreur de type "File not found".

---

## Exécution du script principal

### Lancer tout le TP (Parties 1 à 4)

Dans un terminal, se placer dans le dossier du projet puis exécuter :

```bash
python tp5_issa_spam.py
```

- **Partie 1 :** chargement des données, exploration, visualisations, vectorisation TF-IDF, découpage train/test (quelques secondes).
- **Partie 2 :** entraînement et évaluation de SVM, KNN, XGBoost, LightGBM (quelques minutes).
- **Partie 3 :** optimisation par RandomizedSearchCV (plusieurs minutes, ne pas interrompre), puis TruncatedSVD et sélection de variables.
- **Partie 4 :** enregistrement des runs MLflow, création et sauvegarde du pipeline, analyse de drift, sauvegarde des métadonnées.

Des fenêtres de graphiques (matplotlib) peuvent s’ouvrir ; les fermer pour poursuivre l’exécution.

**Important :** ne pas interrompre le script pendant la Partie 3 (RandomizedSearchCV), sous peine d’arrêt avant la fin et sans génération du pipeline ni des fichiers de sortie.

### Fichiers produits par l’exécution

- **spam_detector_pipeline.pkl** : pipeline (TF-IDF + modèle) sauvegardé pour réutilisation.
- **model_info_v1.json** : métadonnées du modèle (version, métriques, hyperparamètres, etc.).
- **mlruns/** et **mlflow.db** : base et dossiers MLflow pour le suivi des expériences.

---

## Consulter les résultats MLflow

Pour visualiser les runs (modèle baseline et modèle optimisé) dans l’interface MLflow :

1. Dans un terminal, depuis le dossier du projet :

```bash
mlflow ui
```

2. Ouvrir un navigateur à l’adresse : **http://localhost:5000**

3. Sélectionner l’expérience **spam_detection_tp5** pour voir les deux runs (xgboost_baseline et xgboost_optimized).

Arrêter le serveur avec Ctrl+C dans le terminal.

---

## Structure du projet

```
Etudiants/
|
|-- README.md                    (ce fichier)
|-- requirements.txt             (dépendances Python)
|-- tp5_issa_spam.py             (script principal du TP)
|-- utils.py                     (fonctions utilitaires : chargement, graphiques, etc.)
|-- enonce_tp5.md                (énoncé détaillé du TP)
|
|-- data/
|   |-- README.md                (instructions pour le dataset)
|   |-- spam.csv                 (à télécharger, non fourni dans le dépôt)
|
|-- spam_detector_pipeline.pkl    (généré après exécution)
|-- model_info_v1.json           (généré après exécution)
|-- mlflow.db                    (base MLflow, générée après exécution)
|-- mlruns/                      (artefacts MLflow, générés après exécution)
```

Les fichiers **tp5_template.py**, **PLAN_CORRECTION.md** et **tp5_techniques_avancees.ipynb** sont optionnels (template ou notes) et ne sont pas nécessaires pour exécuter le TP.

---

## Livrables du TP

- **tp5_issa_spam.py** : script Python complété.
- **spam_detector_pipeline.pkl** : pipeline de détection de spam (texte en entrée, prédiction en sortie).
- **Rapport MLflow** : captures d’écran ou export des runs (baseline et optimisé), obtenus via `mlflow ui`.
- **README** : ce fichier, décrivant l’installation et l’exécution du projet.

---

## Auteur

Nom et prénom à indiquer pour le rendu (ex. dans l’en-tête du script ou du README).
