'''EDA'''
from textaugment import EDA
import random

eda_augmentor = EDA()

def apply_eda(text, prob=0.7):
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

'''
Usage
processed_train_data = preprocess_text({
    'text': train_df['text'].tolist(),
    'label': train_df['label'].tolist()
}, augment_func=apply_eda)
encoded_train_dataset = Dataset.from_dict(processed_train_data)
'''

'''Back translation using Google Translate'''
from deep_translator import GoogleTranslator
import random
import time

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

'''
Usage
preprocessed_train_data = preprocess_text({
    'text': train_df['text'].tolist(),
    'label': train_df['label'].tolist()
}, augment_func=apply_back_translation_google, batch_before=True)
encoded_train_dataset = Dataset.from_dict(processed_train_data)
'''

'''Back translation using MarianMT'''
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ProcessPoolExecutor
import random

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

'''
Usage
preprocessed_train_data = preprocess_text({
    'text': train_df['text'].tolist(),
    'label': train_df['label'].tolist()
}, augment_func=apply_back_translation_marian, batch_before=True)
encoded_train_dataset = Dataset.from_dict(processed_train_data)
'''
