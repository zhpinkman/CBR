dataset="data/final_data"
dataset_mod=${dataset//"/"/_}

for feature in "text" "explanations" "structure" "counter" "goals"; do
    echo "Feature: $feature"

    for ratio_of_source_used in 0.7 1.0; do

        CUDA_VISIBLE_DEVICES=3,4,5 python -m cbr_analyser.case_retriever.transformers.simcse_similarity_calculations \
            --feature ${feature} \
            --source_file "${dataset}/train.csv" \
            --target_file "${dataset}/split.csv" \
            --output_file "cache/${dataset_mod}/simcse_similarities_${feature}_split_ratio_${ratio_of_source_used}.joblib" \
            --ratio_of_source_used ${ratio_of_source_used}

    done
done
# done

# conda deactivate
