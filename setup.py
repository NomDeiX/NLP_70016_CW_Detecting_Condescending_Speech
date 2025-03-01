import os
import urllib.request
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib import request
from collections import Counter
from ast import literal_eval
from simpletransformers.classification import ClassificationModel, ClassificationArgs,MultiLabelClassificationModel, MultiLabelClassificationArgs
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
if cuda_available:
  import tensorflow as tf
  # Get the GPU device name.
  device_name = tf.test.gpu_device_name()
  # The device name should look like the following:
  if device_name == '/device:GPU:0':
      print('Found GPU at: {}'.format(device_name))
  else:
      raise SystemError('GPU device not found')

import os
import urllib.request

# Define required dataset files
files_to_download = {
    # Main dataset files from CRLala's GitHub
    "dontpatronizeme_pcl.tsv": "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv",
    "dontpatronizeme_categories.tsv": "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_categories.tsv",

    # Train/dev splits from Perez-AlmendrosC's GitHub
    "train_semeval_parids-labels.csv": "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv",
    "dev_semeval_parids-labels.csv": "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv"
}

# Download each file if missing
for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"✔ {filename} already exists.")

# Verify that all files exist
missing_files = [f for f in files_to_download.keys() if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(f"❌ Missing files: {missing_files}. Please check the download URLs.")
else:
    print("✅ All required files are available!")

# Download dont_patronize_me module
module_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
module_name = module_url.split('/')[-1]
print(f'Fetching {module_name}')
#with open("file_1.txt") as f1, open("file_2.txt") as f2
with request.urlopen(module_url) as f, open(module_name,'w') as outf:
  a = f.read()
  outf.write(a.decode('utf-8'))




from dont_patronize_me import DontPatronizeMe

# Load dataset
dpm = DontPatronizeMe('.', '.')
dpm.load_task1()
dpm.load_task2(return_one_hot=True)

import os
from IPython.display import display

# Load paragraph IDs
tr_ids = pd.read_csv('train_semeval_parids-labels.csv')
te_ids = pd.read_csv('dev_semeval_parids-labels.csv')

# Convert IDs to string
tr_ids.par_id = tr_ids.par_id.astype(str)
te_ids.par_id = te_ids.par_id.astype(str)

# Extract dataset
data = dpm.train_task1_df



# Ensure necessary imports
from sklearn.model_selection import train_test_split

# Split dataset while maintaining class balance
train_df, temp_df = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, stratify=temp_df['label'], random_state=42)

# Print dataset sizes
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


import re
import string
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import nltk
nltk.download('punkt_tab', quiet=True)  # Attempt if the error insists
nltk.download('punkt', force=True)      # Force re-download core tokenizer data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


punctuation_table = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()
