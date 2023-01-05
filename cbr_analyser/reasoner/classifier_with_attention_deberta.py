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
from transformers import (TrainerCallback,
                          Trainer, TrainingArguments, DebertaModel, DebertaPreTrainedModel, DebertaTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput
from collections import OrderedDict
# os.environ["WANDB_MODE"] = "dryrun"


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # StableDropout only calls this function when training.
        train = True
        # TODO: We should check if the opset_version being used to export
        # is > 12 here, but there's no good way to do that. As-is, if the
        # opset_version < 12, export will fail with a CheckerError.
        # Once https://github.com/pytorch/pytorch/issues/78391 is fixed, do something like:
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        return symbolic_opset12.dropout(g, input, dropout_p, train)


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size,
                               config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


checkpoint_for_adapter = "cross-encoder/nli-deberta-base"

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


class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        self.attention = MultiheadAttention(
            self.deberta.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_cbr: Optional[torch.LongTensor] = None,
        attention_mask_cbr: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]

        outputs_cbr = self.deberta(
            input_ids_cbr,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask_cbr,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer_cbr = outputs_cbr[0]

        encoder_layer, _ = self.attention(
            query=encoder_layer,
            key=encoder_layer_cbr,
            value=encoder_layer_cbr
        )

        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(
                                label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


def create_augmented_case(row, config, similar_cases: List[str]):
    if config.feature in ['text', 'explanations', 'goals']:
        augmented_case = row['text']
        for similar_case in similar_cases:
            augmented_case += f' {config.sep_token} ' + similar_case
    elif config.feature in ['structure', 'counterfactual']:
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
    outputs_dict['note'] = 'deberta_model_with_attention_check_cbr_different_features_for_retrieval'
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
        tokenizer = DebertaTokenizer.from_pretrained(checkpoint_for_adapter)

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

        model = DebertaForSequenceClassification.from_pretrained(
            checkpoint_for_adapter, num_labels=len(list(label_encoder.classes_)), ignore_mismatched_sizes=True)

        # print('Model loaded!')

        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            output_dir=f"./cbr_deberta_logical_fallacy_classification_{config.data_dir.replace('/', '_')}",
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
            "values": [-1e7, 0.5, 0.8]
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
            'values': [1e-2]
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
    wandb.agent(sweep_id, do_train_process, count=6)
