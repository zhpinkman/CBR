
dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"

for num_cases in 2 3 4 5 1
do


for feature in "counter" "text" "explanations" "structure" "goals"
do

for ratio_of_source_used in 1.0 0.4 0.7 0.1 
do

WANDB_MODE="dryrun" CUDA_VISIBLE_DEVICES=1 python -m cbr_analyser.reasoner.classifier_with_attention_distilbert \
    --data_dir ${dataset} \
    --feature ${feature} \
    --num_cases ${num_cases} \
    --ratio_of_source_used ${ratio_of_source_used}

done

done

done