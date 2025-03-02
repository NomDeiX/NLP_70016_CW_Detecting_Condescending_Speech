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
import sys
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ProcessPoolExecutor
from deep_translator import GoogleTranslator
import time



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

def apply_eda(text, prob=0.55):
   # Ensure text is valid
    if not isinstance(text, str) or text.strip() == "":
        return text  # Return unchanged if empty or NaN

    num_words = len(text.split())

    # Apply each augmentation with a 70% probability
    if random.random() < prob:
        text = eda_augmentor.synonym_replacement(text)  # Synonym Replacement
    if random.random() < prob:
        text = eda_augmentor.random_insertion(text)  # Random Insertion
    if num_words >= 2 and random.random() < prob:
        text = eda_augmentor.random_swap(text)  # Random Swap
    if random.random() < prob:
        text = eda_augmentor.random_deletion(text)  # Random Deletion

    return text



intermediate_langs_google = ['fr', 'de', 'es', 'it', 'ru']

# Create translation dictionaries
translators_google = {
    lang: {
        'to_lang': GoogleTranslator(source="en", target=lang),  # English → Language
        'from_lang': GoogleTranslator(source=lang, target="en")  # Language → English
    }
    for lang in intermediate_langs_google
}

def back_translate_batch_google(text_batch, lang):
    try:
        # Translate to intermediate language
        intermediate_texts = translators_google[lang]['to_lang'].translate_batch(text_batch)

        # Small delay to prevent rate limiting
        time.sleep(0.2)

        # Translate back to English
        back_translated_texts = translators_google[lang]['from_lang'].translate_batch(intermediate_texts)
        return back_translated_texts

    except Exception as e:
        print(f"Batch translation error ({lang}): {e}")
        return text_batch

def apply_back_translation_google(texts, labels, prob=0.7):
    # Select texts for back-translation
    selected_texts = []
    selected_labels = []
    assigned_languages = []

    for text, label in zip(texts, labels):
        if isinstance(text, str) and text.strip() and random.random() < prob:
            selected_texts.append(text)
            selected_labels.append(label)
            assigned_languages.append(random.choice(intermediate_langs_google))

    if not selected_texts:
        return [], []  # No back-translation needed

    # Group texts by language for batch processing
    language_batches = {lang: [] for lang in intermediate_langs_google}
    language_label_batches = {lang: [] for lang in intermediate_langs_google}

    for text, label, lang in zip(selected_texts, selected_labels, assigned_languages):
        language_batches[lang].append(text)
        language_label_batches[lang].append(label)

    # Step 3: Process each language batch sequentially
    augmented_texts = []
    augmented_labels = []

    for lang in intermediate_langs_google:
        if language_batches[lang]:
            translated_batch = back_translate_batch_google(language_batches[lang], lang)
            augmented_texts.extend(translated_batch)
            augmented_labels.extend(language_label_batches[lang])  # Keep label order aligned

    return augmented_texts, augmented_labels


target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)

intermediate_langs_marian = ['fr', 'pt', 'es', 'it']

def translate_marian(texts, model, tokenizer, lang):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if lang == "en" else f">>{lang}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return translated_texts

def back_translate_batch_marian(texts, lang):
    # Translate from source to target language
    intermediate_texts = translate_marian(texts, target_model, target_tokenizer, lang)

    # Translate from target language back to source language
    back_translated_texts = translate_marian(intermediate_texts, en_model, en_tokenizer, "en")

    return back_translated_texts

def process_language_marian(language_batches, language_label_batches, lang):
    if language_batches[lang]:
        translated_batch = back_translate_batch_marian(language_batches[lang], lang)
        return translated_batch, language_label_batches[lang]
    return [], []

def apply_back_translation_marian(texts, labels, prob=0.7):
    # Select texts for back-translation
    selected_texts = []
    selected_labels = []
    assigned_languages = []

    for text, label in zip(texts, labels):
        if isinstance(text, str) and text.strip() and random.random() < prob:
            selected_texts.append(text)
            selected_labels.append(label)
            assigned_languages.append(random.choice(intermediate_langs_marian))

    if not selected_texts:
        return [], []  # No back-translation needed

    # Group texts by language for batch processing
    language_batches = {lang: [] for lang in intermediate_langs_marian}
    language_label_batches = {lang: [] for lang in intermediate_langs_marian}

    for text, label, lang in zip(selected_texts, selected_labels, assigned_languages):
        language_batches[lang].append(text)
        language_label_batches[lang].append(label)

    # Step 3: Process each language batch sequentially
    augmented_texts = []
    augmented_labels = []

    # Run all languages in parallel
    with ProcessPoolExecutor(max_workers=len(intermediate_langs_marian)) as executor:
        future_results = {executor.submit(process_language_marian, language_batches, language_label_batches, lang): lang for lang in intermediate_langs_marian}

    for future in future_results:
        translated_texts, corresponding_labels = future.result()
        augmented_texts.extend(translated_texts)
        augmented_labels.extend(corresponding_labels)

    return augmented_texts, augmented_labels


model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_text(phrases_batched, augment_func=None, batch_before=False):
    original_texts = phrases_batched['text']
    original_labels = phrases_batched['label']

    if augment_func is not None and batch_before is True:
        augmented_texts, augmented_labels = augment_func(original_texts, original_labels)
        original_texts.extend(augmented_texts)
        original_labels.extend(augmented_labels)

    preprocessed_texts = []
    labels = []

    for text, label in zip(original_texts, original_labels):
        # Expand contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)

        # Convert to lowercase
        text = text.lower()

        #  Remove punctuation
        text = text.translate(punctuation_table)

        # Tokenization
        tokens = word_tokenize(text)

        #  Remove stopwords
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

        # Apply lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Remove duplicate words (consider if needed)
        unique_tokens = list(dict.fromkeys(lemmatized_tokens))
        cleaned_text = ' '.join(unique_tokens)

        preprocessed_texts.append(cleaned_text)
        labels.append(label)

        # Apply data augmentation if needed
        if augment_func is not None and batch_before is False:
            augmented_text = augment_func(cleaned_text)
            preprocessed_texts.append(augmented_text)
            labels.append(label)

    # Tokenize the preprocessed texts and add labels
    tokenized = tokenizer(preprocessed_texts, truncation=True, padding=True)
    tokenized["label"] = labels
    return tokenized



f1_metric = evaluate.load("f1")

num_labels = 2 
task = "patronizing-language-detection"

dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

columns_to_remove = dataset_train.column_names
augmentation_type = sys.argv[1]

if augmentation_type == "eda":
    augment_func = apply_eda
    batch_before = False
elif augmentation_type == "backtranslation_marian":
    augment_func = apply_back_translation_marian
    batch_before = True
elif augmentation_type == "backtranslation_google":
    augment_func = apply_back_translation_google
    batch_before = True     

model_name = "DebertaV2"
encoded_train_dataset = dataset_train.map(preprocess_text, batched=True, remove_columns=columns_to_remove, fn_kwargs={"augment_func": augment_func, "batch_before": batch_before})
encoded_val_dataset = dataset_val.map(preprocess_text, batched=True, remove_columns=columns_to_remove, fn_kwargs={"augment_func": augment_func, "batch_before": batch_before})
encoded_test_dataset = dataset_test.map(preprocess_text, batched=True, remove_columns=columns_to_remove, fn_kwargs={"augment_func": augment_func, "batch_before": batch_before})


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
        group=f"data_aug_{augmentation_type}",
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


