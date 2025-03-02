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
import re
import string
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import urllib.request
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
import optuna
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import evaluate
from evaluate import load
import numpy as np
from transformers import AutoTokenizer
from IPython.display import display
from textaugment import EDA
import random
from sklearn.model_selection import train_test_split
import wandb
from dotenv import load_dotenv



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


# Load API keys
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


# for tracking hyper parameters
wandb.login(key=wandb_api_key)



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

# Load paragraph IDs
tr_ids = pd.read_csv('train_semeval_parids-labels.csv')
te_ids = pd.read_csv('dev_semeval_parids-labels.csv')

# Convert IDs to string
tr_ids.par_id = tr_ids.par_id.astype(str)
te_ids.par_id = te_ids.par_id.astype(str)

# Extract dataset
data = dpm.train_task1_df


# Split dataset while maintaining class balance
train_df, temp_df = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.33, stratify=temp_df['label'], random_state=42)

# Print dataset sizes
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



nltk.download('punkt_tab', quiet=True)  # Attempt if the error insists
nltk.download('punkt', force=True)      # Force re-download core tokenizer data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


punctuation_table = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()



eda_augmentor = EDA()

def apply_eda(text):
   # Ensure text is valid
    if not isinstance(text, str) or text.strip() == "":
        return text  # Return unchanged if empty or NaN

    num_words = len(text.split())

    # Apply each augmentation with a 40% probability
    if random.random() < 0.4:
        text = eda_augmentor.synonym_replacement(text)  # Synonym Replacement
    if random.random() < 0.4:
        text = eda_augmentor.random_insertion(text)  # Random Insertion
    if num_words >= 2 and random.random() < 0.4:
        text = eda_augmentor.random_swap(text)  # Random Swap
    if random.random() < 0.4:
        text = eda_augmentor.random_deletion(text)  # Random Deletion

    return text


model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_text(phrases_batched):
    preprocessed_phrases = []
    for text in phrases_batched['text']:
        # Expand contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)

        # Convert to lowercase
        text = text.lower()

        #  Remove punctuation
        text = text.translate(punctuation_table)

        # Apply EDA
        # text = apply_eda(text)

        # Tokenization
        tokens = word_tokenize(text)

        #  Remove stopwords
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

        # Apply lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Remove duplicate words (consider if needed)
        unique_tokens = list(dict.fromkeys(lemmatized_tokens))
        preprocessed_phrases.append(' '.join(unique_tokens))

    return tokenizer(preprocessed_phrases, truncation=True, padding=True)



f1_metric = evaluate.load("f1")

num_labels = 2 
task = "patronizing-language-detection"

dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

columns_to_remove = [col for col in dataset_train.column_names if col != 'label']

model_name = "DebertaV2"
encoded_train_dataset = dataset_train.map(preprocess_text, batched=True, remove_columns=columns_to_remove)
encoded_val_dataset = dataset_val.map(preprocess_text, batched=True, remove_columns=columns_to_remove)
encoded_test_dataset = dataset_test.map(preprocess_text, batched=True, remove_columns=columns_to_remove)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    result = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return result


def objective(trial):

    hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.0, 0.5)
    attention_probs_dropout_prob = trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.5)
    # initializer_range = trial.suggest_float("initializer_range", 0.001, 0.1, log=True) #maybe include this
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32]) #4
    batch_size = 4
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 6)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)


    config = AutoConfig.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        # initializer_range=initializer_range,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
)
    wandb.init(
        entity="mihnea-savin-imperial-college-london",
        project="my-awesome-project",
        config=config.to_dict(),
        reinit=True  # ensures a new run is started for each trial
    )


    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
)
    trainer.train()

    eval_result = trainer.evaluate()
    wandb.finish()

    return eval_result["eval_loss"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(trial.value)
print(trial.params)


