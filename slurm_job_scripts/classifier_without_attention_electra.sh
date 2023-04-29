dataset="data/finegrained_with_structures_explanations"
echo "Dataset: $dataset"

for i in 1 2; do

    for num_cases in 2; do

        for feature in "text" "counter"; do

            for ratio_of_source_used in 0.1; do

                WANDB_MODE="dryrun" CUDA_VISIBLE_DEVICES=5,6,7 python -m cbr_analyser.reasoner.classifier_without_attention_electra \
                    --data_dir ${dataset} \
                    --feature ${feature} \
                    --num_cases ${num_cases} \
                    --ratio_of_source_used ${ratio_of_source_used}

            done

        done

    done
done
