# Scripts


This folder contains all the scripts to run the training and evaluation of the models. The scripts are for both training of the models and also for extraction and pre computation of the similarity look up tables. 

For the models, as we used `ELECTRA`, `RoBERTa`, `BERT`, and `DistilBERT`, we have a script for each of them. 


For the similarity look up tables, we have a script that based on the feature and ratio of the data to use, will compute the similarity look up tables for all the datasets.

The only thing to remeber when running the similarity pre computation job, `simcse_similarity_calculations.sh`, is to have a folder with the exact name of the dataset directory in the `cache` directory with a prefix of `data`. For example, if you want to compute the similarity look up tables for the `final_data` dataset, you should have a folder named `data_final_data` in the `cache` directory.

    A quick note, you can specify the gpu that should be available to the model in the beginning of the script, as it's already set for different scripts for different gpus. The same applies for the WANDB initialization. it can be set to online or offline mode. We use WANDB for all the hyperparameters tuning and logging of the results.

After executing each training job, the results of the training, the predictions of the model, the most similar examples extracted for each data point and all the settings that were applied when training the model will be saved in the `predictions_dir` directory fed to the training file. However, you can skip passing this argument as its default value is set to a subdirectory called `predictions/all` in `cache` directory. Be sure to have the respective output subdirectory in the `cache` directory as it will be used to save the results of the training.