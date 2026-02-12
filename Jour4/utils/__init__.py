"""
Package utils pour le TP4 - Détection de Fraude Bancaire
"""

from .feature_engineering import (
    create_temporal_features,
    create_amount_features,
    create_interaction_features,
    create_aggregation_features,
    create_deviation_features,
    create_all_features,
    create_features,
    FeatureEngineer,
    get_top_correlated_features,
)

from .predict import FraudDetector, predict_fraud

__all__ = [
    'create_temporal_features',
    'create_amount_features',
    'create_interaction_features',
    'create_aggregation_features',
    'create_deviation_features',
    'create_all_features',
    'create_features',
    'FeatureEngineer',
    'get_top_correlated_features',
    'FraudDetector',
    'predict_fraud'
]

