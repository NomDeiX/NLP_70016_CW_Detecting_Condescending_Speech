from transformers import AutoTokenizer

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


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
