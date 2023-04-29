from torch.nn import MultiheadAttention
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datetime import datetime
import wandb
from cbr_analyser.case_retriever.retriever import (
    Retriever,
    SentenceTransformerRetriever,
    SimCSE_Retriever,
    Empathy_Retriever,
)
import argparse
import joblib
from torch.optim import Adam
from IPython import embed
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers.activations import get_activation
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (
    TrainerCallback,
    Trainer,
    TrainingArguments,
    ElectraModel,
    ElectraPreTrainedModel,
    ElectraTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# os.environ["WANDB_MODE"] = "dryrun"


checkpoint_for_adapter = "howey/electra-base-mnli"

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction",
]


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        # although BERT uses tanh here, it seems Electra authors used gelu here
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.attention = MultiheadAttention(
            self.electra.config.hidden_size, num_heads=8, batch_first=True
        )
        self.classifier = ElectraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


def create_augmented_case(row, config, similar_cases: List[str]):
    if config.feature in ["text", "explanations", "goals"]:
        augmented_case = row["text"]
        for similar_case in similar_cases:
            augmented_case += f" {config.sep_token} " + similar_case
    elif config.feature in ["structure", "counter"]:
        augmented_case = row["text"]
        for similar_case in similar_cases:
            augmented_case += f" {config.sep_token} {row[config.feature]} {config.sep_token} {similar_case}"
    return augmented_case


def augment_with_similar_cases(
    df: pd.DataFrame, retrievers: List[Retriever], config
) -> pd.DataFrame:
    all_similar_cases = []
    all_augmented_cases = []
    all_similar_cases_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        row_similar_cases = []
        row_similar_cases_labels = []
        for retriever in retrievers:
            try:
                similar_cases_with_labels = retriever.retrieve_similar_cases(
                    case=row[config.feature],
                    num_cases=config.num_cases,
                    threshold=config.cbr_threshold,
                )

                row_similar_cases.extend(
                    [
                        case_label["similar_case"]
                        for case_label in similar_cases_with_labels
                    ]
                )
                row_similar_cases_labels.extend(
                    [
                        case_label["similar_case_label"]
                        for case_label in similar_cases_with_labels
                    ]
                )

            except Exception as e:
                print(e)
                embed()

        augmented_case = create_augmented_case(row, config, row_similar_cases)

        all_augmented_cases.append(augmented_case)
        all_similar_cases.append(row_similar_cases)
        all_similar_cases_labels.append(row_similar_cases_labels)

    df["augmented_cases"] = all_augmented_cases
    df["similar_cases"] = all_similar_cases
    df["similar_cases_labels"] = all_similar_cases_labels
    return df


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def save_results(
    config, label_encoder, predictions, predictions_climate, test_df, climate_df
):
    now = datetime.today().isoformat()

    # run_name = wandb.run.name
    outputs_dict = {}
    outputs_dict[
        "note"
    ] = "electra_model_without_attention_check_cbr_different_features_for_retrieval_after_rebuttals"
    outputs_dict["label_encoder"] = label_encoder
    outputs_dict["meta"] = dict(config)
    # outputs_dict['run_name'] = run_name
    outputs_dict["predictions"] = predictions._asdict() if predictions else None
    outputs_dict["predictions_climate"] = (
        predictions_climate._asdict() if predictions_climate else None
    )

    outputs_dict["text"] = test_df["text"].tolist()
    outputs_dict["augmented_cases"] = test_df["augmented_cases"].tolist()
    outputs_dict["similar_cases"] = test_df["similar_cases"].tolist()
    outputs_dict["similar_cases_labels"] = test_df["similar_cases_labels"].tolist()

    outputs_dict["text_climate"] = climate_df["text"].tolist()
    outputs_dict["augmented_cases_climate"] = climate_df["augmented_cases"].tolist()
    outputs_dict["similar_cases_climate"] = climate_df["similar_cases"].tolist()
    outputs_dict["similar_cases_labels_climate"] = climate_df[
        "similar_cases_labels"
    ].tolist()

    file_name = os.path.join(config.predictions_dir, f"outputs_dict__{now}.joblib")

    joblib.dump(outputs_dict, file_name)


def do_train_process(config=None):
    with wandb.init(config=config):
        config = wandb.config

        if config.eval_only:
            tokenizer = ElectraTokenizer.from_pretrained(config.model_dir)
            model = ElectraForSequenceClassification.from_pretrained(config.model_dir)
        else:
            tokenizer = ElectraTokenizer.from_pretrained(checkpoint_for_adapter)
            model = ElectraForSequenceClassification.from_pretrained(
                checkpoint_for_adapter,
                num_labels=len(list(label_encoder.classes_)),
                classifier_dropout=config.classifier_dropout,
                ignore_mismatched_sizes=True,
            )

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        climate_df = pd.read_csv(os.path.join(config.data_dir, "climate_test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        climate_df = climate_df[~climate_df["label"].isin(bad_classes)]

        print("using cbr")

        retrievers_to_use = []
        for retriever_str in config.retrievers:
            if retriever_str == "simcse":
                simcse_retriever = SimCSE_Retriever(config)
                retrievers_to_use.append(simcse_retriever)
            elif retriever_str == "empathy":
                empathy_retriever = Empathy_Retriever(config)
                retrievers_to_use.append(empathy_retriever)
            elif retriever_str.startswith("sentence-transformers"):
                sentence_transformers_retriever = SentenceTransformerRetriever(config)
                retrievers_to_use.append(sentence_transformers_retriever)

        dfs_to_process = [train_df, dev_df, test_df, climate_df]
        for df in dfs_to_process:
            df = augment_with_similar_cases(df, retrievers_to_use, config)
        try:
            del retrievers_to_use
            del simcse_retriever
            del empathy_retriever
            del coarse_retriever
        except:
            pass

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["label"])

        train_df["label"] = label_encoder.transform(train_df["label"])
        dev_df["label"] = label_encoder.transform(dev_df["label"])
        test_df["label"] = label_encoder.transform(test_df["label"])
        climate_df["label"] = label_encoder.transform(climate_df["label"])

        if config.data_dir == "data/bigbench":
            dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df),
                    "eval": Dataset.from_pandas(dev_df),
                    "test": Dataset.from_pandas(test_df),
                }
            )
        else:
            dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df),
                    "eval": Dataset.from_pandas(dev_df),
                    "test": Dataset.from_pandas(test_df),
                    "climate": Dataset.from_pandas(climate_df),
                }
            )

        def process(batch):
            inputs_cbr = tokenizer(
                batch["augmented_cases"], truncation=True, padding="max_length"
            )
            return {
                "input_ids": inputs_cbr["input_ids"],
                "attention_mask": inputs_cbr["attention_mask"],
                "labels": batch["label"],
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset["train"].column_names
        )

        print("Model loaded!")

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"models/cbr_electra_wo_attention_logical_fallacy_classification_{config.data_dir.replace('/', '_')}",
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

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # callbacks=[PrinterCallback]
        )

        if not config.eval_only:
            print("Start the training ...")
            trainer.train()
            trainer.save_model(
                f"models/cbr_electra_wo_attention_logical_fallacy_classification_{config.data_dir.replace('/', '_')}"
            )

        predictions = trainer.predict(tokenized_dataset["test"])
        predictions_climate = trainer.predict(tokenized_dataset["climate"])

        save_results(
            config, label_encoder, predictions, predictions_climate, test_df, climate_df
        )


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_only", help="Whether to only evaluate the model", action="store_true"
    )

    parser.add_argument("--model_dir", help="Model directory", type=str)

    parser.add_argument(
        "--data_dir",
        help="Train input file path",
        type=str,
        default="data/new_finegrained",
    )
    parser.add_argument(
        "--predictions_dir",
        help="Predictions output file path",
        default="cache/predictions/all",
        type=str,
    )

    parser.add_argument(
        "--retrievers_similarity_func",
        help="Checkpoint namespace",
        type=str,
        default="simcse",
    )

    parser.add_argument(
        "--num_cases", help="Number of cases in CBR", type=int, default=1
    )

    parser.add_argument(
        "--feature", help="Feature to use for retrieval", type=str, default="text"
    )

    parser.add_argument("--mode", help="Mode", type=str, default="cbr")

    parser.add_argument(
        "--ratio_of_source_used",
        help="Ratio of training data used for the case database",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    sweep_config = {
        "method": "grid",
    }

    metric = {"name": "eval/f1", "goal": "maximize"}

    sweep_config["metric"] = metric

    parameters_dict = {
        "eval_only": {"values": [args.eval_only]},
        "model_dir": {"values": [args.model_dir]},
        "ratio_of_source_used": {"values": [args.ratio_of_source_used]},
        "checkpoint_for_adapter": {"values": [checkpoint_for_adapter]},
        "sep_token": {"values": ["[SEP]"]},
        "retrievers": {"values": [[args.retrievers_similarity_func]]},
        "feature": {"values": [args.feature]},
        "num_cases": {"values": [args.num_cases]},
        "cbr_threshold": {"values": [-1e7]},
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
        sweep_config,
        project="CBR framework with different entities considered for similarity retrieval",
    )
    wandb.agent(sweep_id, do_train_process, count=1)
