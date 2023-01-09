dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"

for num_cases in 1 2 3
do

for ratio_of_source_used in 0.4 0.7 0.1 1.0
do

for feature in "explanations" "goals" "counter"
do

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=4 python -m cbr_analyser.reasoner.classifier_with_attention_roberta \
    --data_dir ${dataset} \
    --feature ${feature} \
    --num_cases ${num_cases} \
    --ratio_of_source_used ${ratio_of_source_used}

done

done

done