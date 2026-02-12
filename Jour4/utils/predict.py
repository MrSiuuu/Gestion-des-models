"""
Module de prédiction pour le modèle de détection de fraude
TP4 - Master 1 Data Engineering

Ce module permet de charger le modèle entraîné et de faire des prédictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class FraudDetector:
    """
    Classe pour la détection de fraude bancaire
    """
    
    def __init__(self, model_path='../models/fraud_detector_v1.joblib', 
                 metadata_path='../models/metadata_v1.joblib'):
        """
        Initialise le détecteur de fraude
        
        Args:
            model_path (str): Chemin vers le modèle sauvegardé
            metadata_path (str): Chemin vers les métadonnées
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata = None
        self.threshold = 0.5  # Seuil par défaut
        
        self.load_model()
    
    def load_model(self):
        """Charge le modèle et les métadonnées"""
        try:
            self.model = joblib.load(self.model_path)
            print(f" Modèle chargé depuis {self.model_path}")
            
            if self.metadata_path.exists():
                self.metadata = joblib.load(self.metadata_path)
                print(f" Métadonnées chargées depuis {self.metadata_path}")
                
                # Utiliser le seuil optimal si disponible
                if 'optimal_threshold' in self.metadata:
                    self.threshold = self.metadata['optimal_threshold']
                    print(f"   Seuil de décision: {self.threshold:.4f}")
            else:
                print("  Métadonnées non trouvées, utilisation du seuil par défaut (0.5)")
                
        except FileNotFoundError as e:
            print(f" Erreur: Modèle non trouvé à {self.model_path}")
            raise e
    
    def predict(self, transaction_data, use_optimal_threshold=True):
        """
        Prédit si une transaction est frauduleuse
        
        Args:
            transaction_data (dict, pd.Series, pd.DataFrame): Données de transaction
            use_optimal_threshold (bool): Utiliser le seuil optimal ou 0.5
        
        Returns:
            dict: Résultat de la prédiction
        """
        # TODO: Implémenter la prédiction
        # 1. Convertir transaction_data en DataFrame si nécessaire
        # 2. Vérifier que toutes les features requises sont présentes
        # 3. Prédire la probabilité
        # 4. Appliquer le seuil
        # 5. Déterminer le niveau de risque
        # 6. Retourner un dictionnaire avec les résultats
        
        # Convertir en DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, pd.Series):
            df = transaction_data.to_frame().T
        elif isinstance(transaction_data, pd.DataFrame):
            df = transaction_data.copy()
        else:
            raise ValueError("Format de données non supporté")
        
        # Prédire la probabilité
        proba = self.model.predict_proba(df)[:, 1]
        
        # Appliquer le seuil
        threshold = self.threshold if use_optimal_threshold else 0.5
        prediction = (proba >= threshold).astype(int)
        
        # Déterminer le niveau de risque
        risk_level = self._get_risk_level(proba[0])
        
        # Résultat
        result = {
            'is_fraud': bool(prediction[0]),
            'probability': float(proba[0]),
            'risk_level': risk_level,
            'threshold_used': threshold,
            'confidence': float(abs(proba[0] - 0.5) * 2)  # 0 = incertain, 1 = très confiant
        }
        
        return result
    
    def predict_batch(self, transactions_df, use_optimal_threshold=True):
        """
        Prédit pour un batch de transactions
        
        Args:
            transactions_df (pd.DataFrame): DataFrame de transactions
            use_optimal_threshold (bool): Utiliser le seuil optimal
        
        Returns:
            pd.DataFrame: DataFrame avec prédictions
        """
        # TODO: Implémenter la prédiction batch
        
        # Prédire les probabilités
        probas = self.model.predict_proba(transactions_df)[:, 1]
        
        # Appliquer le seuil
        threshold = self.threshold if use_optimal_threshold else 0.5
        predictions = (probas >= threshold).astype(int)
        
        # Créer le DataFrame de résultats
        results_df = transactions_df.copy()
        results_df['fraud_probability'] = probas
        results_df['is_fraud'] = predictions
        results_df['risk_level'] = [self._get_risk_level(p) for p in probas]
        
        return results_df
    
    def _get_risk_level(self, probability):
        """
        Détermine le niveau de risque basé sur la probabilité
        
        Args:
            probability (float): Probabilité de fraude (0-1)
        
        Returns:
            str: Niveau de risque (low/medium/high/critical)
        """
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        elif probability < 0.85:
            return 'high'
        else:
            return 'critical'
    
    def explain_prediction(self, transaction_data):
        """
        Explique une prédiction (placeholder pour SHAP)
        
        Args:
            transaction_data: Données de transaction
        
        Returns:
            dict: Explication de la prédiction
        """
        # TODO: Implémenter avec SHAP (bonus)
        result = self.predict(transaction_data)
        
        explanation = {
            'prediction': result,
            'message': "Explication détaillée non implémentée (nécessite SHAP)",
            'top_features': "À implémenter avec Feature Importance"
        }
        
        return explanation
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle
        
        Returns:
            dict: Informations du modèle
        """
        if self.metadata:
            return self.metadata
        else:
            return {
                'model_loaded': self.model is not None,
                'threshold': self.threshold,
                'metadata_available': False
            }


def predict_fraud(transaction_data, model_path='../models/fraud_detector_v1.joblib'):
    """
    Fonction utilitaire pour prédire rapidement
    
    Args:
        transaction_data: Données de transaction
        model_path (str): Chemin vers le modèle
    
    Returns:
        dict: Résultat de la prédiction
    """
    detector = FraudDetector(model_path)
    return detector.predict(transaction_data)


if __name__ == "__main__":
    # Test du module
    print("="*60)
    print("MODULE DE PRÉDICTION - DÉTECTION DE FRAUDE")
    print("="*60)
    
    # Exemple d'utilisation
    print("\n Exemple d'utilisation:")
    print("""
    from utils.predict import FraudDetector
    
    # Initialiser le détecteur
    detector = FraudDetector()
    
    # Prédire pour une transaction
    transaction = {
        'Time': 12345,
        'V1': -1.5,
        'V2': 2.3,
        # ... autres features
        'Amount': 150.0
    }
    
    result = detector.predict(transaction)
    print(result)
    # {'is_fraud': True, 'probability': 0.87, 'risk_level': 'high', ...}
    
    # Prédire pour un batch
    results_df = detector.predict_batch(transactions_df)
    """)
    
    print("\n Module chargé avec succès!")

