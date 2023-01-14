dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"


# WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=7 python -m cbr_analyser.reasoner.classifier_with_attention_roberta_baseline \
#     --data_dir ${dataset}


# WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=7 python -m cbr_analyser.reasoner.classifier_with_attention_bert_baseline \
#     --data_dir ${dataset}



WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=4 python -m cbr_analyser.reasoner.classifier_with_attention_bart_baseline \
    --data_dir ${dataset}