"""
Tests unitaires - Détection de fraude
TP4 - Master 1 Data Engineering
"""

import unittest
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_engineering import create_features, get_top_correlated_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detector_v1.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')


def _get_X_sample(n=10):
    """Construit un X avec les mêmes colonnes que le pipeline (feature engineering + dummies)."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, nrows=n)
    top3 = get_top_correlated_features(df, 'Class', 5)[:3]
    df_fe = create_features(df.copy(), top3)
    df_enc = pd.get_dummies(df_fe, columns=['period', 'amount_bin'], drop_first=True)
    return df_enc.drop(columns=['Class'])


class TestFraudDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Modèle non trouvé: {MODEL_PATH}")

    def test_model_loading(self):
        """Test de chargement du modèle"""
        model = joblib.load(MODEL_PATH)
        self.assertIsNotNone(model)

    def test_prediction_shape(self):
        """Test que les prédictions ont la bonne forme"""
        X = _get_X_sample(10)
        if X is None:
            self.skipTest("Dataset non trouvé")
        model = joblib.load(MODEL_PATH)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(X))

    def test_prediction_values(self):
        """Test que les probabilités sont dans [0, 1]"""
        X = _get_X_sample(10)
        if X is None:
            self.skipTest("Dataset non trouvé")
        model = joblib.load(MODEL_PATH)
        probas = model.predict_proba(X)[:, 1]
        self.assertTrue(np.all((probas >= 0) & (probas <= 1)))

    def test_prediction_time(self):
        """Test que la prédiction est rapide (< 1s pour 10 transactions)"""
        X = _get_X_sample(10)
        if X is None:
            self.skipTest("Dataset non trouvé")
        model = joblib.load(MODEL_PATH)
        start = time.time()
        model.predict(X)
        elapsed_ms = (time.time() - start) * 1000
        self.assertLess(elapsed_ms, 1000)


if __name__ == '__main__':
    unittest.main()
