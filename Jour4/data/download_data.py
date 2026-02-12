"""
Script de téléchargement du dataset Credit Card Fraud Detection
Dataset Kaggle : https://www.kaggle.com/mlg-ulb/creditcardfraud

IMPORTANT : Ce dataset nécessite un compte Kaggle et l'API Kaggle configurée.

Installation :
1. pip install kaggle
2. Créer un compte Kaggle
3. Télécharger votre API token : https://www.kaggle.com/settings/account
4. Placer kaggle.json dans ~/.kaggle/

Alternative : Téléchargement manuel depuis Kaggle puis placer creditcard.csv ici.
"""

import os
import sys
import zipfile
from pathlib import Path

def download_with_kaggle():
    """Télécharge le dataset via l'API Kaggle"""
    try:
        import kaggle
        print(" Téléchargement du dataset via Kaggle API...")
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='.',
            unzip=True
        )
        
        print(" Dataset téléchargé avec succès !")
        return True
        
    except ImportError:
        print(" Module 'kaggle' non installé.")
        print("   Installez-le avec : pip install kaggle")
        return False
        
    except Exception as e:
        print(f" Erreur lors du téléchargement : {e}")
        print("\n Instructions :")
        print("   1. Créez un compte sur https://www.kaggle.com")
        print("   2. Allez dans Account > Create New API Token")
        print("   3. Placez kaggle.json dans ~/.kaggle/")
        print("   4. chmod 600 ~/.kaggle/kaggle.json")
        return False

def check_dataset():
    """Vérifie si le dataset existe déjà"""
    csv_path = Path('creditcard.csv')
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f" Dataset déjà présent : {csv_path} ({size_mb:.1f} MB)")
        return True
    return False

def get_dataset_info():
    """Affiche les informations sur le dataset"""
    try:
        import pandas as pd
        
        if not Path('creditcard.csv').exists():
            print(" Dataset non trouvé.")
            return
        
        print("\n Informations sur le dataset :")
        df = pd.read_csv('creditcard.csv')
        
        print(f"   - Nombre de transactions : {len(df):,}")
        print(f"   - Nombre de features : {df.shape[1]}")
        print(f"   - Nombre de fraudes : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
        print(f"   - Nombre de légitimes : {(df['Class']==0).sum():,}")
        print(f"   - Taille du fichier : {Path('creditcard.csv').stat().st_size / (1024*1024):.1f} MB")
        print(f"\n   Colonnes : {', '.join(df.columns.tolist())}")
        
    except ImportError:
        print("  Pandas non installé, impossible d'afficher les infos.")
    except Exception as e:
        print(f" Erreur : {e}")

def main():
    print("="*60)
    print("  TÉLÉCHARGEMENT DATASET - DÉTECTION DE FRAUDE BANCAIRE")
    print("="*60)
    
    # Vérifier si le dataset existe déjà
    if check_dataset():
        get_dataset_info()
        print("\n Rien à faire, le dataset est prêt !")
        return
    
    print("\n Le dataset n'est pas présent localement.")
    print("\n Deux options :")
    print("   1. Téléchargement automatique via Kaggle API (recommandé)")
    print("   2. Téléchargement manuel depuis Kaggle")
    
    choice = input("\nChoisir l'option (1 ou 2) : ").strip()
    
    if choice == '1':
        if download_with_kaggle():
            get_dataset_info()
        else:
            print("\n  Téléchargement automatique échoué.")
            print("   Utilisez l'option 2 (téléchargement manuel).")
    
    elif choice == '2':
        print("\n Instructions pour le téléchargement manuel :")
        print("   1. Allez sur : https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("   2. Cliquez sur 'Download' (connexion requise)")
        print("   3. Décompressez creditcard.csv")
        print("   4. Placez-le dans ce dossier (TD/data/)")
        print("\n   Puis relancez ce script pour vérifier.")
    
    else:
        print(" Option invalide.")

if __name__ == "__main__":
    main()

