from torch.nn import MultiheadAttention
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datetime import datetime
import wandb
from cbr_analyser.case_retriever.retriever import (
    Retriever, SentenceTransformerRetriever, SimCSE_Retriever, Empathy_Retriever)
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
from transformers import (TrainerCallback, Trainer, TrainingArguments,
                          RobertaForSequenceClassification, RobertaTokenizer)


checkpoint_for_adapter = "cross-encoder/nli-roberta-base"

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        with open('logs.txt', 'w') as f:
            f.write(f"{str(logs)}\n")


def save_results(config, label_encoder, predictions, predictions_climate, test_df):
    now = datetime.today().isoformat()

    outputs_dict = {}
    outputs_dict['note'] = 'roberta_model_with_attention_check_cbr_different_features_for_retrieval_baseline'
    outputs_dict['label_encoder'] = label_encoder
    outputs_dict["meta"] = dict(config)
    # outputs_dict['run_name'] = run_name
    outputs_dict['predictions'] = predictions._asdict()
    outputs_dict['predictions_climate'] = predictions_climate._asdict()
    outputs_dict['text'] = test_df['text'].tolist()
    outputs_dict['augmented_cases'] = []
    outputs_dict['similar_cases'] = []
    outputs_dict['similar_cases_labels'] = []

    file_name = os.path.join(
        config.predictions_dir,
        f"outputs_dict__{now}.joblib"
    )

    joblib.dump(outputs_dict, file_name)


def do_train_process(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = RobertaTokenizer.from_pretrained(
            checkpoint_for_adapter)

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        climate_df = pd.read_csv(os.path.join(
            config.data_dir, "climate_test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        climate_df = climate_df[~climate_df["label"].isin(bad_classes)]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['label'])

        train_df['label'] = label_encoder.transform(train_df['label'])
        dev_df['label'] = label_encoder.transform(dev_df['label'])
        test_df['label'] = label_encoder.transform(test_df['label'])
        climate_df['label'] = label_encoder.transform(climate_df['label'])

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'eval': Dataset.from_pandas(dev_df),
            'test': Dataset.from_pandas(test_df),
            'climate': Dataset.from_pandas(climate_df)
        })

        def process(batch):
            inputs = tokenizer(
                batch["text"], truncation=True, padding='max_length'
            )
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': batch['label']
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        model = RobertaForSequenceClassification.from_pretrained(
            checkpoint_for_adapter, num_labels=len(list(label_encoder.classes_)), classifier_dropout=config.classifier_dropout, ignore_mismatched_sizes=True)

        # print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"./cbr_roberta_logical_fallacy_classification_{config.data_dir.replace('/', '_')}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy='steps',
            report_to="wandb",
            # auto_find_batch_size=True,
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['eval'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # callbacks=[PrinterCallback]
        )

        # print('Start the training ...')
        trainer.train()

        predictions = trainer.predict(tokenized_dataset['test'])
        predictions_climate = trainer.predict(tokenized_dataset['climate'])

        save_results(config, label_encoder, predictions,
                     predictions_climate, test_df)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', help="Train input file path", type=str, default="data/new_finegrained"
    )
    parser.add_argument('--predictions_dir', help="Predictions output file path",
                        default="cache/predictions/all", type=str)

    parser.add_argument(
        '--retrievers_similarity_func', help="Checkpoint namespace", type=str, default="simcse")

    parser.add_argument(
        '--num_cases', help="Number of cases in CBR", type=int, default=1)

    parser.add_argument(
        '--feature', help="Feature to use for retrieval", type=str, default="text")

    parser.add_argument('--mode', help="Mode", type=str, default="cbr")

    parser.add_argument('--ratio_of_source_used',
                        help="Ratio of training data used for the case database", type=float, default=1.0)

    args = parser.parse_args()

    sweep_config = {
        'method': 'grid',
    }

    metric = {
        'name': 'eval/f1',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'ratio_of_source_used': {
            'values': [args.ratio_of_source_used]
        },
        'checkpoint_for_adapter': {
            'values': [checkpoint_for_adapter]
        },
        'sep_token': {
            'values': ['[SEP]']
        },
        'retrievers': {
            "values": [
                [args.retrievers_similarity_func]
                # ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'],
                # ['sentence-transformers/paraphrase-MiniLM-L6-v2'],
                # ['sentence-transformers/all-MiniLM-L12-v2'],
                # ['sentence-transformers/all-MiniLM-L6-v2'],
                # ['simcse'],
                # ['empathy'],
                # ["simcse", "empathy"],
                # ["simcse"],
                # ["empathy"]
            ]
        },
        'feature': {
            'values': [args.feature]
        },
        'num_cases': {
            # "values": [4] if args.data_dir == "data/new_finegrained" else [1] if args.data_dir == "data/finegrained" else [1] if args.data_dir == "data/coarsegrained" else [3]
            # "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            "values": [args.num_cases]
        },
        'cbr_threshold': {
            "values": [-1e7]
            # "values": [0.5]
            # "values": [-10000000] if args.data_dir == "data/new_finegrained" else [-10000000] if args.data_dir == "data/finegrained" else [-10000000] if args.data_dir == "data/coarsegrained" else [0.5]
        },
        'data_dir': {
            "values": [args.data_dir]
        },
        'predictions_dir': {
            "values": [args.predictions_dir]
        },
        'batch_size': {
            "values": [16]
        },
        'learning_rate': {
            'values': [8.447927580802138e-05]
            # 'distribution': 'uniform',
            # 'min': 1e-5,
            # 'max': 1e-4
            # 'min': 3e-5 if args.data_dir == "data/finegrained" else 1e-5,
            # 'max': 6e-5 if args.data_dir == "data/finegrained" else 1e-4,
            # "values": [3.120210415844665e-05] if args.data_dir == "data/new_finegrained" else [7.484147412800621e-05] if args.data_dir == "data/finegrained" else [7.484147412800621e-05] if args.data_dir == "data/coarsegrained" else [5.393991227358502e-06]
        },
        "num_epochs": {
            "values": [10]
        },
        "classifier_dropout": {
            # "values": [0.1, 0.3, 0.8]
            'values': [0.1]
            # "values": [0.8] if args.data_dir == "data/new_finegrained" else [0.3] if args.data_dir == "data/finegrained" else [0.3] if args.data_dir == "data/coarsegrained" else [0.1]
        },
        'weight_decay': {
            'values': [0.04962960561110768]
            # 'distribution': 'uniform',
            # 'min': 1e-4,
            # 'max': 1e-1
            # "values": [0.07600643653465429] if args.data_dir == "data/new_finegrained" else [0.00984762513370293] if args.data_dir == "data/finegrained" else [0.00984762513370293] if args.data_dir == "data/coarsegrained" else [0.022507698737927326]
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(
        sweep_config, project="CBR framework with different entities considered for similarity retrieval")
    wandb.agent(sweep_id, do_train_process, count=1)
