import re
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from dont_patronize_me import DontPatronizeMe

# Minimal text preprocessing for baselines
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def train_evaluate_model(x_train, x_test, y_train, y_test, vectorizer, model_name):
    vectorizer.fit(x_train)
    x_train_vec = vectorizer.transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train_vec, y_train)
    y_pred = model.predict(x_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    print(classification_report(y_test, y_pred))

    # Store misclassified & correctly classified examples
    results_df = pd.DataFrame({
        "Text": x_test,
        "True Label": y_test,
        "Predicted Label": y_pred
    })

    misclassified = results_df[results_df["True Label"] != results_df["Predicted Label"]]
    correctly_classified = results_df[results_df["True Label"] == results_df["Predicted Label"]]

    # Save results for later analysis
    misclassified.to_csv(f"misclassified_{model_name}.csv", index=False)
    correctly_classified.to_csv(f"correctly_classified_{model_name}.csv", index=False)

    print(f"\nSaved misclassified and correctly classified examples for {model_name}.")

if __name__ == "__main__":
    # Load dataset
    dpm = DontPatronizeMe('.', '.')
    dpm.load_task1()
    df = dpm.train_task1_df

    df['text'] = df['text'].apply(preprocess_text)

    x = df['text']
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=42)

    print("Training BoW + Logistic Regression...")
    train_evaluate_model(x_train, x_test, y_train, y_test, CountVectorizer(), "BoW")

    print("Training TF-IDF + Logistic Regression...")
    train_evaluate_model(x_train, x_test, y_train, y_test, TfidfVectorizer(), "TF-IDF")
