from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
import optuna
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import evaluate
from evaluate import load
import numpy as np

f1_metric = evaluate.load("f1")

num_labels = 2 
task = "patronizing-language-detection"

dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)


model_name = "DebertaV2"
encoded_train_dataset = dataset_train.map(preprocess_text, batched=True)
encoded_val_dataset = dataset_val.map(preprocess_text, batched=True)
encoded_test_dataset = dataset_test.map(preprocess_text, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    result = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return result


def objective(trial):

    hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.0, 0.5)
    attention_probs_dropout_prob = trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.5)
    # initializer_range = trial.suggest_float("initializer_range", 0.001, 0.1, log=True) #maybe include this
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32]) #4
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

    return eval_result["eval_loss"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(trial.value)
print(trial.params)


