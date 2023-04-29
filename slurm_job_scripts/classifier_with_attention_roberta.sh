dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"

for num_cases in 2; do

    for ratio_of_source_used in 0.1; do

        for feature in "explanations"; do

            WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -m cbr_analyser.reasoner.classifier_with_attention_roberta \
                --data_dir ${dataset} \
                --feature ${feature} \
                --num_cases ${num_cases} \
                --ratio_of_source_used ${ratio_of_source_used} \
                --eval_only \
                --model_dir "models/cbr_roberta_logical_fallacy_classification_data_finegrained_with_structures_explanations_feature_explanations"

        done

    done

done
