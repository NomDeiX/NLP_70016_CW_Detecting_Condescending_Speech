from textaugment import EDA
import random

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

