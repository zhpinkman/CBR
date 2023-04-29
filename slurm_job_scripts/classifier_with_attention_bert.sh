dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"

for num_cases in 1; do

    for ratio_of_source_used in 0.1; do

        for feature in "text"; do

            WANDB_MODE="dryrun" CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -m cbr_analyser.reasoner.classifier_with_attention_bert \
                --data_dir ${dataset} \
                --feature ${feature} \
                --num_cases ${num_cases} \
                --ratio_of_source_used ${ratio_of_source_used} \
                --eval_only \
                --model_dir "models/cbr_bert_logical_fallacy_classification_data_finegrained_with_structures_explanations"

        done

    done

done
