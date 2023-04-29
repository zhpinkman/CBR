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


def do_train_process(config=None):
    with wandb.init(config=config):
        config = wandb.config
        tokenizer = RobertaTokenizer.from_pretrained("cross-encoder/nli-roberta-base")

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["label"])

        train_df["label"] = label_encoder.transform(train_df["label"])
        dev_df["label"] = label_encoder.transform(dev_df["label"])
        test_df["label"] = label_encoder.transform(test_df["label"])

        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "eval": Dataset.from_pandas(dev_df),
                "test": Dataset.from_pandas(test_df),
            }
        )

        def process(batch):
            inputs = tokenizer(batch["text"], truncation=True, padding="max_length")
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": batch["label"],
            }

        tokenizer = RobertaTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset["train"].column_names
        )

        model = RobertaForSequenceClassification.from_pretrained(
            "cross-encoder/nli-roberta-base",
            num_labels=len(list(label_encoder.classes_)),
            classifier_dropout=config.classifier_dropout,
            ignore_mismatched_sizes=True,
        )

        print("Model loaded!")

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir="./cbr_roberta_logical_fallacy_classification",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy="steps",
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

        print("Start the training ...")
        trainer.train()

        predictions = trainer.predict(tokenized_dataset["test"])

        run_name = wandb.run.name
        outputs_dict = {}
        outputs_dict["note"] = "best_hps_final_baseline_best_ps"
        outputs_dict["label_encoder"] = label_encoder
        outputs_dict["meta"] = dict(config)
        outputs_dict["run_name"] = run_name
        outputs_dict["predictions"] = predictions._asdict()
        outputs_dict["text"] = test_df["text"].tolist()

        now = datetime.today().isoformat()
        file_name = os.path.join(
            config.predictions_dir, f"outputs_dict_{run_name}_{now}.joblib"
        )
        print(file_name)
        joblib.dump(outputs_dict, file_name)
        print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Classification Model for Logical Fallacy Detection and having a baseline"
    )

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
        "retrievers": {"values": [["simcse"]]},
        "num_cases": {"values": [1]},
        "cbr_threshold": {"values": [-1e7, 0.5]},
        "data_dir": {"values": [args.data_dir]},
        "predictions_dir": {"values": [args.predictions_dir]},
        "batch_size": {"values": [16]},
        "learning_rate": {"values": [8.447927580802138e-05]},
        "num_epochs": {"values": [8]},
        "classifier_dropout": {"values": [0.1]},
        "weight_decay": {"values": [0.04962960561110768]},
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(
        sweep_config, project="Baseline Finder with CBR and different retrievers"
    )
    wandb.agent(sweep_id, do_train_process, count=18)
