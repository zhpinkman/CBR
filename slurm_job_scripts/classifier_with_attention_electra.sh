for dataset in "data/finegrained" "data/finegrained_augmented" "data/coarsegrained" "data/coarsegrained_augmented"
do

for num_cases in 1
do

for feature in "text"
do

for ratio_of_source_used in 1.0
do

WANDB_MODE="dryrun" CUDA_VISIBLE_DEVICES=7 python -m cbr_analyser.reasoner.classifier_with_attention_electra_wo_climate \
    --data_dir ${dataset} \
    --feature ${feature} \
    --num_cases ${num_cases} \
    --ratio_of_source_used ${ratio_of_source_used}

done

done

done

done