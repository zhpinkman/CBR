from torch.nn import MultiheadAttention
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datetime import datetime
import wandb
from cbr_analyser.case_retriever.retriever import (
    Retriever, SentenceTransformerRetriever, SimCSE_Retriever, Empathy_Retriever)
import argparse
import joblib
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
from transformers.activations import get_activation, ACT2FN
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (TrainerCallback, PretrainedConfig,
                          Trainer, TrainingArguments, DistilBertModel, DistilBertPreTrainedModel, DistilBertTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.attention = MultiheadAttention(
            self.distilbert.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_cbr: Optional[torch.Tensor] = None,
        attention_mask_cbr: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)

        distilbert_output_cbr = self.distilbert(
            input_ids=input_ids_cbr,
            attention_mask=attention_mask_cbr,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state_cbr = distilbert_output_cbr[0]  # (bs, seq_len, dim)

        final_hidden_state, _ = self.attention(
            query=hidden_state,
            key=hidden_state_cbr,
            value=hidden_state_cbr
        )

        pooled_output = final_hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


checkpoint_for_adapter = "typeform/distilbert-base-uncased-mnli"

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


def create_augmented_case(row, config, similar_cases: List[str]):
    if config.feature in ['text', 'explanations', 'goals']:
        augmented_case = row['text']
        for similar_case in similar_cases:
            augmented_case += f' {config.sep_token} ' + similar_case
    elif config.feature in ['structure', 'counter']:
        augmented_case = row['text']
        for similar_case in similar_cases:
            augmented_case += f" {config.sep_token} {row[config.feature]} {config.sep_token} {similar_case}"
    return augmented_case


def augment_with_similar_cases(df: pd.DataFrame, retrievers: List[Retriever], config) -> pd.DataFrame:
    all_similar_cases = []
    all_augmented_cases = []
    all_similar_cases_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):

        row_similar_cases = []
        row_similar_cases_labels = []
        for retriever in retrievers:
            try:

                similar_cases_with_labels = retriever.retrieve_similar_cases(
                    case=row[config.feature], num_cases=config.num_cases, threshold=config.cbr_threshold
                )

                row_similar_cases.extend(
                    [case_label['similar_case']
                        for case_label in similar_cases_with_labels]
                )
                row_similar_cases_labels.extend(
                    [case_label['similar_case_label']
                        for case_label in similar_cases_with_labels]
                )

            except Exception as e:
                print(e)

        augmented_case = create_augmented_case(row, config, row_similar_cases)

        all_augmented_cases.append(augmented_case)
        all_similar_cases.append(row_similar_cases)
        all_similar_cases_labels.append(row_similar_cases_labels)

    df['augmented_cases'] = all_augmented_cases
    df['similar_cases'] = all_similar_cases
    df['similar_cases_labels'] = all_similar_cases_labels
    return df


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_ids_cbr=inputs["input_ids_cbr"],
            attention_mask_cbr=inputs["attention_mask_cbr"],
        )

        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def save_results(config, label_encoder, predictions, predictions_climate, test_df):
    now = datetime.today().isoformat()

    # run_name = wandb.run.name
    outputs_dict = {}
    outputs_dict['note'] = 'distilbert_model_with_attention_check_cbr_different_features_for_retrieval'
    outputs_dict['label_encoder'] = label_encoder
    outputs_dict["meta"] = dict(config)
    # outputs_dict['run_name'] = run_name
    outputs_dict['predictions'] = predictions._asdict()
    outputs_dict['predictions_climate'] = predictions_climate._asdict()
    outputs_dict['text'] = test_df['text'].tolist()
    outputs_dict['augmented_cases'] = test_df['augmented_cases'].tolist()
    outputs_dict['similar_cases'] = test_df['similar_cases'].tolist()
    outputs_dict['similar_cases_labels'] = test_df['similar_cases_labels'].tolist()

    file_name = os.path.join(
        config.predictions_dir,
        f"outputs_dict__{now}.joblib"
    )

    joblib.dump(outputs_dict, file_name)


def do_train_process(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = DistilBertTokenizer.from_pretrained(checkpoint_for_adapter)

        train_df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))
        dev_df = pd.read_csv(os.path.join(config.data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))
        climate_df = pd.read_csv(os.path.join(
            config.data_dir, "climate_test.csv"))

        train_df = train_df[~train_df["label"].isin(bad_classes)]
        dev_df = dev_df[~dev_df["label"].isin(bad_classes)]
        test_df = test_df[~test_df["label"].isin(bad_classes)]
        climate_df = climate_df[~climate_df["label"].isin(bad_classes)]

        # print('using cbr')

        retrievers_to_use = []
        for retriever_str in config.retrievers:
            if retriever_str == 'simcse':
                simcse_retriever = SimCSE_Retriever(config)
                retrievers_to_use.append(simcse_retriever)
            elif retriever_str == 'empathy':
                empathy_retriever = Empathy_Retriever(config)
                retrievers_to_use.append(empathy_retriever)
            elif retriever_str.startswith('sentence-transformers'):
                sentence_transformers_retriever = SentenceTransformerRetriever(
                    config)
                retrievers_to_use.append(sentence_transformers_retriever)

        dfs_to_process = [train_df, dev_df, test_df, climate_df]
        for df in dfs_to_process:
            df = augment_with_similar_cases(
                df, retrievers_to_use, config
            )
        try:
            del retrievers_to_use
            del simcse_retriever
            del empathy_retriever
            del coarse_retriever
        except:
            pass

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df['label'])

        train_df['label'] = label_encoder.transform(train_df['label'])
        dev_df['label'] = label_encoder.transform(dev_df['label'])
        test_df['label'] = label_encoder.transform(test_df['label'])
        climate_df['label'] = label_encoder.transform(climate_df['label'])

        if config.data_dir == 'data/bigbench':
            dataset = DatasetDict({
                'train': Dataset.from_pandas(train_df),
                'eval': Dataset.from_pandas(dev_df),
                'test': Dataset.from_pandas(test_df),
            })
        else:
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
            inputs_cbr = tokenizer(
                batch["augmented_cases"], truncation=True, padding='max_length'
            )
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'input_ids_cbr': inputs_cbr['input_ids'],
                'attention_mask_cbr': inputs_cbr['attention_mask'],
                'labels': batch['label']
            }

        tokenized_dataset = dataset.map(
            process, batched=True, remove_columns=dataset['train'].column_names)

        model = DistilBertForSequenceClassification.from_pretrained(
            checkpoint_for_adapter, num_labels=len(list(label_encoder.classes_)), ignore_mismatched_sizes=True)

        # print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"./cbr_distilbert_logical_fallacy_classification_{config.data_dir.replace('/', '_')}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            logging_steps=200,
            evaluation_strategy='steps',
            report_to="wandb",
            auto_find_batch_size=True,
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

        trainer = CustomTrainer(
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

    parser.add_argument('--ratio_of_source_used',
                        help="Ratio of training data used for the case database", type=float)

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
            "values": [-1e7, 0.5]
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
            'values': [4.5e-5]
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
    wandb.agent(sweep_id, do_train_process, count=4)