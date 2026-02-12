"""
Module de Feature Engineering pour la détection de fraude bancaire
TP4 - Master 1 Data Engineering

Ce module contient les fonctions de création de features avancées.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def create_temporal_features(df):
    """
    Crée des features temporelles à partir de la colonne Time
    Formules: hour = (Time/3600)%24, day = (Time/86400)%2,
    hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24), period = f(hour)
    """
    df = df.copy()
    if 'Time' not in df.columns:
        return df
    df['hour'] = (df['Time'] / 3600) % 24
    df['day'] = (df['Time'] / 86400) % 2
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['period'] = df['hour'].apply(get_period)
    return df


def create_amount_features(df):
    """
    Crée des features sur les montants: log, sqrt, squared, cubed,
    is_zero_amount, amount_bin (bins 0,10,50,100,500,inf).
    """
    df = df.copy()
    if 'Amount' not in df.columns:
        return df
    df['amount_log'] = np.log(df['Amount'] + 1)
    df['amount_sqrt'] = np.sqrt(df['Amount'])
    df['amount_squared'] = df['Amount'] ** 2
    df['amount_cubed'] = df['Amount'] ** 3
    df['is_zero_amount'] = (df['Amount'] == 0).astype(int)
    df['amount_bin'] = pd.cut(
        df['Amount'],
        bins=[0, 10, 50, 100, 500, np.inf],
        labels=['micro', 'small', 'medium', 'large', 'xlarge']
    )
    return df


def create_interaction_features(df, top_features=None):
    """
    Crée des features d'interaction: V_i*Amount (top 3 PCA), amount_per_hour, time_amount_ratio.
    """
    df = df.copy()
    if top_features is None:
        top_features = ['V1', 'V2', 'V3']
    for feature in top_features:
        if feature in df.columns and 'Amount' in df.columns:
            df[f'{feature}_amount'] = df[feature] * df['Amount']
    if 'Amount' in df.columns and 'hour' in df.columns:
        df['amount_per_hour'] = df['Amount'] / (df['hour'] + 1)
    if 'Time' in df.columns and 'Amount' in df.columns:
        df['time_amount_ratio'] = df['Time'] / (df['Amount'] + 1)
    return df


def create_aggregation_features(df, v_features=None):
    """
    Crée des features d'agrégation: pca_sum_top5, pca_mean_top5, pca_std (V1..V28).
    """
    df = df.copy()
    if v_features is None:
        v_features = [f'V{i}' for i in range(1, 6)]
    available_top5 = [f for f in v_features if f in df.columns]
    if available_top5:
        df['pca_sum_top5'] = df[available_top5].sum(axis=1)
        df['pca_mean_top5'] = df[available_top5].mean(axis=1)
    v_all = [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
    if v_all:
        df['pca_std'] = df[v_all].std(axis=1)
    return df


def create_deviation_features(df, top_features=None):
    """
    Features d'écart (z-score absolu): deviation_Vi = |Vi - mean(Vi)| / std(Vi)
    pour les 3 features PCA les plus corrélées.
    """
    df = df.copy()
    if top_features is None:
        top_features = ['V1', 'V2', 'V3']
    for f in top_features:
        if f in df.columns:
            m, s = df[f].mean(), df[f].std()
            df[f'deviation_{f}'] = np.abs(df[f] - m) / (s + 1e-8)
    return df


def create_all_features(df, top_corr_features=None):
    """
    Applique toutes les transformations (au minimum 15 nouvelles features).
    Alias attendu par le sujet: create_features(df).
    """
    df = df.copy()
    df = create_temporal_features(df)
    df = create_amount_features(df)
    df = create_interaction_features(df, top_corr_features)
    df = create_aggregation_features(df)
    df = create_deviation_features(df, top_corr_features[:3] if top_corr_features else None)
    return df


def create_features(df, top_corr_features=None):
    """
    Crée toutes les features (livrable sujet). Alias de create_all_features.
    """
    return create_all_features(df, top_corr_features)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer Scikit-Learn pour le Feature Engineering
    Compatible avec les Pipelines
    """
    
    def __init__(self, top_corr_features=None):
        """
        Args:
            top_corr_features (list): Features à utiliser pour interactions
        """
        self.top_corr_features = top_corr_features
    
    def fit(self, X, y=None):
        """Fit (pas de paramètres à apprendre)"""
        return self
    
    def transform(self, X):
        """
        Applique les transformations de feature engineering
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            pd.DataFrame: Features transformées
        """
        return create_all_features(X, self.top_corr_features)


# Fonctions utilitaires

def get_period(hour):
    """
    Retourne la période de la journée
    
    Args:
        hour (float): Heure (0-23)
    
    Returns:
        str: Période (night/morning/afternoon/evening)
    """
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def get_top_correlated_features(df, target_col='Class', n=10):
    """
    Retourne les N features les plus corrélées avec la cible
    
    Args:
        df (pd.DataFrame): DataFrame
        target_col (str): Nom de la colonne cible
        n (int): Nombre de features à retourner
    
    Returns:
        list: Liste des features
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonne {target_col} non trouvée")
    
    # Calculer les corrélations
    correlations = df.corr()[target_col].abs()
    
    # Exclure la cible elle-même
    correlations = correlations.drop(target_col)
    
    # Trier et retourner le top N
    top_features = correlations.nlargest(n).index.tolist()
    
    return top_features


if __name__ == "__main__":
    # Test du module
    print("Module Feature Engineering chargé avec succès !")
    print("\nFonctions disponibles:")
    print("- create_temporal_features()")
    print("- create_amount_features()")
    print("- create_interaction_features()")
    print("- create_aggregation_features()")
    print("- create_all_features()")
    print("- FeatureEngineer (Transformer)")
    print("- get_top_correlated_features()")

