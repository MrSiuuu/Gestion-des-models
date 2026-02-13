# Dataset - Détection de Spam SMS

## Source

**Dataset Kaggle**: SMS Spam Collection Dataset
- **Lien**: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Alternative**: https://www.kaggle.com/datasets/team-ai/spam-text-message-classification

## Description

Ce dataset contient 5,574 messages SMS en anglais, tagués comme étant soit du spam (messages non sollicités) soit du ham (messages légitimes). 

Le dataset est **déséquilibré**, avec environ 13.4% de spam et 86.6% de messages légitimes (ham).

## Téléchargement

### Option 1: Via Kaggle API (recommandé)

```bash
# Installer l'API Kaggle
pip install kaggle

# Configurer vos credentials Kaggle (créer un token sur kaggle.com/account)
# Placer le fichier kaggle.json dans ~/.kaggle/

# Télécharger le dataset
kaggle datasets download -d uciml/sms-spam-collection-dataset
unzip sms-spam-collection-dataset.zip -d .
```

### Option 2: Téléchargement manuel

1. Aller sur https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Cliquer sur "Download" (nécessite un compte Kaggle gratuit)
3. Décompresser le fichier `spam.csv` dans ce dossier `data/`

### Option 3: Dataset alternatif

Si vous préférez une alternative:
```bash
kaggle datasets download -d team-ai/spam-text-message-classification
```

## Structure du Dataset

Le dataset contient **2 colonnes**:

### Variables
- **v1** (ou **label**): Type de message
  - `spam`: Message spam (non sollicité)
  - `ham`: Message légitime
  
- **v2** (ou **text/message**): Contenu textuel du SMS

## Caractéristiques

- **Nombre de messages**: 5,574
- **Spam**: ~747 (13.4%)
- **Ham (légitime)**: ~4,827 (86.6%)
- **Langue**: Anglais
- **Format**: Texte brut

### Exemples de messages

**Spam** :
- "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121"

**Ham** :
- "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
- "Ok lar... Joking wif u oni..."

## Particularités

### Déséquilibre de Classes
Le ratio spam/ham est déséquilibré (13.4% spam). Cela nécessite:
- Des métriques adaptées (F1-Score, ROC-AUC, Precision-Recall)
- Des techniques de rééquilibrage ou de pondération (`class_weight='balanced'`)
- Une validation croisée stratifiée

### Traitement du Texte (NLP)
Les messages SMS nécessitent un preprocessing spécifique:
- **Vectorisation**: TF-IDF ou CountVectorizer pour convertir texte en features numériques
- **Nettoyage**: Suppression ponctuation, lowercase, stop words
- **Tokenization**: Découpage en mots/tokens

### Taille
Le fichier CSV fait environ **500 KB** (très léger). Le dataset complet peut être chargé en quelques secondes.

## Utilisation dans le TP

```python
import pandas as pd
from utils import load_spam_dataset

# Option 1: Utiliser la fonction fournie
df = load_spam_dataset('data/spam.csv')

# Option 2: Charger manuellement
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# Vérifier la structure
print(df.shape)
print(df['v1'].value_counts())  # ou df['label'].value_counts()
```

## Références

- **Source Originale**: UCI Machine Learning Repository
- **Créateurs**: Tiago A. Almeida et José María Gómez Hidalgo
- **Publication**: "SMS Spam Collection v.1" - http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

## Notes pour les Étudiants

⚠️ **Important**: Téléchargez le dataset **avant** de commencer le TP pour ne pas perdre de temps pendant la session.

💡 **Astuce**: Ce dataset est léger (500 KB), le téléchargement est très rapide.

📝 **NLP**: Ce TP introduit le traitement du langage naturel (NLP) avec la vectorisation TF-IDF.

🔒 **Données publiques**: Ce dataset est open-source et peut être utilisé librement à des fins pédagogiques et de recherche.
