import wandb
from datetime import datetime
import argparse
import joblib
import os
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (
    RobertaForSequenceClassification,
    TrainingArguments,
    RobertaTokenizer,
    Trainer,
)


os.environ["WANDB_MODE"] = "dryrun"


bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction",
]


def save_results(
    config, label_encoder, predictions, predictions_climate, test_df, climate_df
):
    now = datetime.today().isoformat()

    # run_name = wandb.run.name
    outputs_dict = {}
    outputs_dict[
        "note"
    ] = "electra_model_with_attention_check_cbr_different_features_for_retrieval"
    outputs_dict["label_encoder"] = label_encoder
    outputs_dict["meta"] = dict(config)
    outputs_dict["predictions"] = predictions._asdict() if predictions else None
    outputs_dict["predictions_climate"] = (
        predictions_climate._asdict() if predictions_climate else None
    )

    outputs_dict["text"] = test_df["text"].tolist()
    outputs_dict["text_climate"] = climate_df["text"].tolist()

    file_name = os.path.join(config.predictions_dir, f"outputs_dict__{now}.joblib")

    joblib.dump(outputs_dict, file_name)


def do_train_process(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        climate_df = pd.read_csv(os.path.join(config.data_dir, "climate_test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        climate_df = climate_df[~climate_df["label"].isin(bad_classes)]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["label"])

        train_df["label"] = label_encoder.transform(train_df["label"])
        dev_df["label"] = label_encoder.transform(dev_df["label"])
        test_df["label"] = label_encoder.transform(test_df["label"])
        climate_df["label"] = label_encoder.transform(climate_df["label"])

        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "eval": Dataset.from_pandas(dev_df),
                "test": Dataset.from_pandas(test_df),
                "climate_test": Dataset.from_pandas(climate_df),
            }
        )

        if config.eval_only:
            tokenizer = RobertaTokenizer.from_pretrained(config.model_dir)
            model = RobertaForSequenceClassification.from_pretrained(config.model_dir)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(
                "cross-encoder/nli-roberta-base"
            )
            model = RobertaForSequenceClassification.from_pretrained(
                "cross-encoder/nli-roberta-base",
                num_labels=len(list(label_encoder.classes_)),
                classifier_dropout=config.classifier_dropout,
                ignore_mismatched_sizes=True,
            )

        def process(batch):
            inputs = tokenizer(batch["text"], truncation=True, padding="max_length")
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": batch["label"],
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset["train"].column_names
        )

        print("Model loaded!")

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"models/cbr_baseline_roberta_{config.data_dir.replace('/', '_')}",
            save_total_limit=2,
            load_best_model_at_end=True,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            save_strategy="steps",
            logging_strategy="steps",
            evaluation_strategy="steps",
            logging_steps=200,
            eval_steps=200,
            save_steps=200,
            report_to="wandb",
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted"
            )
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        if not config.eval_only:
            print("Start the training ...")
            trainer.train()

            trainer.save_model(
                f"models/cbr_baseline_roberta_{config.data_dir.replace('/', '_')}"
            )

        predictions = trainer.predict(tokenized_dataset["test"])
        predictions_climate = trainer.predict(tokenized_dataset["climate_test"])

        save_results(
            config, label_encoder, predictions, predictions_climate, test_df, climate_df
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Classification Model for Logical Fallacy Detection and having a baseline"
    )

    parser.add_argument(
        "--eval_only", help="Whether to only evaluate the model", action="store_true"
    )

    parser.add_argument("--model_dir", help="Model directory", type=str)

    parser.add_argument(
        "--data_dir", help="Train input file path", type=str, default="data/finegrained"
    )
    parser.add_argument(
        "--predictions_dir",
        help="Predictions output file path",
        default="cache/predictions/all",
        type=str,
    )

    args = parser.parse_args()

    sweep_config = {
        "method": "random",
    }

    metric = {"name": "eval/f1", "goal": "maximize"}

    sweep_config["metric"] = metric

    parameters_dict = {
        "eval_only": {"values": [args.eval_only]},
        "model_dir": {"values": [args.model_dir]},
        "retrievers": {"values": [["simcse"]]},
        "num_cases": {"values": [1]},
        "cbr_threshold": {"values": [-1e7, 0.5]},
        "data_dir": {"values": [args.data_dir]},
        "predictions_dir": {"values": [args.predictions_dir]},
        "batch_size": {"values": [16]},
        "learning_rate": {"values": [8.447927580802138e-05]},
        "num_epochs": {"values": [6]},
        "classifier_dropout": {"values": [0.1]},
        "weight_decay": {"values": [0.04962960561110768]},
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(
        sweep_config, project="Baseline Finder with CBR and different retrievers"
    )
    wandb.agent(sweep_id, do_train_process, count=1)
