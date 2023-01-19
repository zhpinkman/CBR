dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"


for num_cases in 1 2 3 4 5
do

for feature in "goals" "text" "explanations" "counter" "structure"
do

for ratio_of_source_used in 1.0
do

WANDB_MODE="dryrun" CUDA_VISIBLE_DEVICES=5 python -m cbr_analyser.reasoner.classifier_with_attention_electra \
    --data_dir ${dataset} \
    --feature ${feature} \
    --num_cases ${num_cases} \
    --ratio_of_source_used ${ratio_of_source_used}

done

done

done