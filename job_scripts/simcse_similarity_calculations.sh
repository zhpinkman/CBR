#!/bin/bash
#SBATCH --job-name=simcse_similarity_calculations
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/logical-fallacy-identification/CBR
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
# echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
# eval "$(conda shell.bash hook)"
# Activate (local) env
# conda activate general

dataset="data/final_data"
dataset_mod=${dataset//"/"/_}

# for split in "train" "dev" "test" "climate_test"; do

for feature in "text" "explanations" "structure" "counter" "goals"; do
    echo "Feature: $feature"

    for ratio_of_source_used in 0.4; do

        CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m cbr_analyser.case_retriever.transformers.simcse_similarity_calculations \
            --feature ${feature} \
            --source_file "${dataset}/train.csv" \
            --target_file "${dataset}/split.csv" \
            --output_file "cache/${dataset_mod}/simcse_similarities_${feature}_split_ratio_${ratio_of_source_used}.joblib" \
            --ratio_of_source_used ${ratio_of_source_used}

    done
done
# done

# conda deactivate
