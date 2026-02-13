"""
Fonctions utilitaires pour le TP5 - Techniques Avancées & MLOps
Master 1 Data Engineering - YNOV Montpellier

Sujet : Détection de Spam SMS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, roc_auc_score
)
from typing import Tuple, Optional


def load_spam_dataset(
    filepath: str = 'data/spam.csv',
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Charge le dataset Kaggle de détection de spam SMS.
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier CSV du dataset Kaggle
    sample_size : int, optional
        Si spécifié, charge seulement les N premières lignes (utile pour tests rapides)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame avec les colonnes 'label' (spam/ham) et 'message' (texte du SMS)
    """
    try:
        # Le dataset peut avoir différents encodings et noms de colonnes
        if sample_size:
            df = pd.read_csv(filepath, encoding='latin-1', nrows=sample_size)
            print(f"✓ Échantillon chargé: {sample_size} messages")
        else:
            df = pd.read_csv(filepath, encoding='latin-1')
            print(f"✓ Dataset complet chargé: {len(df)} messages")
        
        # Normaliser les noms de colonnes (peut varier selon la version)
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
            df = df[['label', 'message']]  # Garder seulement les 2 colonnes importantes
        elif 'type' in df.columns and 'text' in df.columns:
            df = df.rename(columns={'type': 'label', 'text': 'message'})
        
        # Informations de base
        n_spam = (df['label'] == 'spam').sum()
        n_ham = (df['label'] == 'ham').sum()
        print(f"  - Messages légitimes (ham): {n_ham} ({n_ham/len(df)*100:.2f}%)")
        print(f"  - Messages spam: {n_spam} ({n_spam/len(df)*100:.2f}%)")
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {filepath} n'existe pas.")
        print("\nVeuillez télécharger le dataset depuis Kaggle:")
        print("https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        print("\nOu utilisez la fonction de téléchargement automatique:")
        print("  kaggle datasets download -d uciml/sms-spam-collection-dataset")
        raise


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matrice de Confusion",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Affiche une matrice de confusion avec un heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Prédictions du modèle
    title : str
        Titre du graphique
    figsize : tuple
        Taille de la figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Afficher aussi le rapport de classification
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=['Ham', 'Spam'],
        digits=4
    ))


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Modèle",
    figsize: Tuple[int, int] = (10, 6)
) -> float:
    """
    Affiche la courbe ROC et calcule l'AUC.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes
    y_proba : array-like
        Probabilités prédites pour la classe positive
    model_name : str
        Nom du modèle pour la légende
    figsize : tuple
        Taille de la figure
        
    Returns:
    --------
    float
        Score AUC-ROC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'{model_name} (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Hasard (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title('Courbe ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Modèle",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Affiche la courbe Precision-Recall.
    Particulièrement utile pour les classes déséquilibrées.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes
    y_proba : array-like
        Probabilités prédites pour la classe positive
    model_name : str
        Nom du modèle pour la légende
    figsize : tuple
        Taille de la figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, label=model_name)
    plt.xlabel('Rappel (Recall)', fontsize=12)
    plt.ylabel('Précision (Precision)', fontsize=12)
    plt.title('Courbe Précision-Rappel', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_models_performance(
    results_dict: dict,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Compare les performances de plusieurs modèles.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire avec nom_modèle: {metric: value}
        Exemple: {'SVM': {'accuracy': 0.95, 'f1': 0.85, 'auc': 0.92}}
    figsize : tuple
        Taille de la figure
    """
    df_results = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Graphique en barres
    df_results.plot(kind='bar', ax=axes[0], rot=45)
    axes[0].set_title('Comparaison des Métriques par Modèle', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modèle', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].legend(title='Métrique', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Heatmap
    sns.heatmap(
        df_results.T,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        ax=axes[1],
        cbar_kws={'label': 'Score'}
    )
    axes[1].set_title('Heatmap des Performances', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Modèle', fontsize=12)
    axes[1].set_ylabel('Métrique', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher aussi le tableau
    print("\n" + "="*60)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("="*60)
    print(df_results.to_string())
    print("\nMeilleur modèle par métrique:")
    for metric in df_results.columns:
        best_model = df_results[metric].idxmax()
        best_score = df_results[metric].max()
        print(f"  - {metric}: {best_model} ({best_score:.4f})")


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Affiche l'importance des features d'un modèle.
    
    Parameters:
    -----------
    feature_names : list
        Liste des noms de features
    importances : array-like
        Scores d'importance
    top_n : int
        Nombre de features à afficher
    figsize : tuple
        Taille de la figure
    """
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features les Plus Importantes', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def detect_data_drift(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature: str,
    bins: int = 30
) -> None:
    """
    Détecte visuellement le drift d'une feature entre train et test.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Données d'entraînement
    test_data : pd.DataFrame
        Données de test/production
    feature : str
        Nom de la feature à analyser
    bins : int
        Nombre de bins pour l'histogramme
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_data[feature], bins=bins, alpha=0.6, label='Train', color='blue', density=True)
    plt.hist(test_data[feature], bins=bins, alpha=0.6, label='Test/Production', color='red', density=True)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.title(f'Distribution de {feature}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(
        [train_data[feature].dropna(), test_data[feature].dropna()],
        labels=['Train', 'Test/Production']
    )
    plt.ylabel(feature, fontsize=12)
    plt.title(f'Boxplot de {feature}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques descriptives
    print("\n" + "="*60)
    print(f"STATISTIQUES DESCRIPTIVES - {feature}")
    print("="*60)
    print("\nTrain:")
    print(train_data[feature].describe())
    print("\nTest/Production:")
    print(test_data[feature].describe())


def save_model_info(model, filepath: str, metadata: Optional[dict] = None) -> None:
    """
    Sauvegarde les informations d'un modèle pour la traçabilité.
    
    Parameters:
    -----------
    model : estimator
        Modèle scikit-learn
    filepath : str
        Chemin pour sauvegarder les informations
    metadata : dict, optional
        Métadonnées supplémentaires (hyperparamètres, métriques, etc.)
    """
    import json
    from datetime import datetime
    
    info = {
        'model_type': type(model).__name__,
        'timestamp': datetime.now().isoformat(),
        'parameters': model.get_params() if hasattr(model, 'get_params') else {},
        'metadata': metadata or {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=4, default=str)
    
    print(f"✓ Informations du modèle sauvegardées dans: {filepath}")


if __name__ == "__main__":
    # Test de chargement du dataset
    print("Test de chargement du dataset Kaggle...")
    print("\nNote: Assurez-vous d'avoir téléchargé le dataset depuis:")
    print("https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("\nTest avec un petit échantillon (si le fichier existe)...")
    
    try:
        df = load_spam_dataset(sample_size=100)
        print(f"\n✓ Test réussi!")
        print(f"Dimensions: {df.shape}")
        print(f"\nColonnes: {list(df.columns)}")
        print(f"\nExemple de message spam:")
        spam_msg = df[df['label']=='spam']['message'].iloc[0] if len(df[df['label']=='spam']) > 0 else "Aucun spam dans l'échantillon"
        print(spam_msg)
    except:
        print("\n⚠️ Dataset non trouvé - normal si vous ne l'avez pas encore téléchargé")
        print("Les fonctions utilitaires sont prêtes à être utilisées!")
