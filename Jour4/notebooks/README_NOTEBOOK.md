# INSTRUCTIONS - Notebook TP4

## Contenu du Notebook

Le notebook `TP4_Detection_Fraude_ETUDIANT.ipynb` est structuré en 5 parties :

### Partie 1 : Exploration & Feature Engineering (1h)
- Chargement et analyse des données
- Analyse du déséquilibre
- Corrélations
- **Création de 8+ nouvelles features**

### Partie 2 : Modélisation Baseline (45min)
- Préparation des données (split, scaling)
- 3 modèles baseline (Logistic, Tree, Random Forest)
- Comparaison des performances

### Partie 3 : Optimisation Avancée (1h30)
- Construction d'un Pipeline ML
- GridSearchCV pour optimiser les hyperparamètres
- Analyse des résultats

### Partie 4 : Analyse & Diagnostic (45min)
- Feature Importance
- Courbes ROC et Precision-Recall
- Learning Curves
- Analyse des erreurs
- Optimisation du seuil de décision

### Partie 5 : Déploiement & Production (30min)
- Validation temporelle (TimeSeriesSplit)
- Sérialisation du modèle
- Test de chargement

## Objectifs de Performance

- **Recall ≥ 0.85** (détecter 85% des fraudes)
- **Precision maximale** (minimiser les faux positifs)
- **PR-AUC > 0.75**

## Questions à Répondre

Le notebook contient 10 questions de réflexion à répondre directement dans les cellules markdown.

## Conseils

1. **Commencez par lire tout le notebook** pour comprendre la structure
2. **Testez chaque cellule** avant de passer à la suivante
3. **Utilisez l'aide-mémoire** (`AIDE_MEMOIRE.md`) pour la syntaxe
4. **Documentez vos choix** dans les questions
5. **Sauvegardez régulièrement** votre travail

## Pour Démarrer

```bash
# Installer les dépendances
pip install -r ../requirements.txt

# Télécharger les données
cd ../data
python download_data.py

# Lancer Jupyter
cd ../notebooks
jupyter notebook TP4_Detection_Fraude_ETUDIANT.ipynb
```

Bon courage ! 

