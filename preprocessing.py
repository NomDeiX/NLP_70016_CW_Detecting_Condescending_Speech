from transformers import AutoTokenizer

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


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
