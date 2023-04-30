This repository contains the code for the paper [Case-Based Reasoning with Language Models for Classification of Logical Fallacies](https://arxiv.org/abs/2301.11879) that was accepted to the IJCAI 2023 conference. The model uses case based reasoning to reinforce language models to identify and categorize logical fallacies more accurately and more explainable. 

### Installation

To install the dependencies, you can use the provided `environment.yml` file to create a conda environment with all the required dependencies. To do so, you can run the following command:

```bash
conda env create --name cbr -f environment.yml
```



In the following, each major directory and the code that is contained in it is explained: 

* [Cache](#cache)

* [Dataset](#dataset)

* [Retriever](#retriever)

* [Adapter](#adapter)

* [Job Scripts](#job-scripts)

* [Crawler](#crawler)


### Cache
Since the most time consuming part of the experiments is computing the look up tables for the retrievers to find similar examples to a new example, its look up tables are saved in `cache` directory. Using different settings, the look up tables and other cached files corresponding to each setting is stored in each associated `joblib` file. Due to the large size of cached files, you can access the zipped file of the cache [here](). After downloading the file, you can unzip it under the cache directory.


### Dataset

The data we used which comes from the original logical fallacy [paper](https://arxiv.org/abs/2202.13758) is in `data` folder, in subdirectories of the `data_without_augmentation` and `final_data`. `data_without_augmentation` directory contains the data that we initially used and then augmented it. It contains different splits such as `train`, `dev`, `test` and `climate_test`. `final_data` directory contains the final data that we used for our analysis. It contains different splits such as `train`, `dev`, `test` and `climate_test`. The only difference between `final_data` and the `data_without_augmentation` is that the `final_data` has all the extra case representation we gathered for our analysis as well as the augmentations we did for the data.


All the logic contained in the project is contained in the `cbr_analyser`. 

### Retriever

All the codes and resources that are used to compute the embeddings as well as similarity look up tables for the retriever component are in `cbr_analyser/case_retriever`. The retriever component is the first stage of the Case-based reasoning pipeline. The retriever component is responsible for finding the most similar cases to a new example. The retriever component uses the `transformers` and `SimCSE` library to compute the embeddings of the examples.

After running the similarity calculations in `transformers` directory, their similarity look up tables are stored in `cache` directory, and further will be used when the reasoner models are trained. Note that these similarities can be computed per case representations and also using different ratios of the case data base. These parameters can be changed in the `job_scripts/simcse_similarity_calculations.sh` script.


### Adapter

Except for the first stage of the Case-based reasoning pipeline that is handled by the retriever and discussed separately in the pervious section, the other sections, namely, the adapter and classifier and their code are in `cbr_analyser/reasoner` directory. Be sure that the similarity look up tables are computed and stored in `cache` directory before running the adapter and classifier (the script for computing the similarities is located in the `job_scripts/simcse_similarity_calculations.sh`). The easiest way to run these models is to use the job scripts discussed in the next section. Different features and different hyperparameters that can be set for running the models are all included in the scripts as arguments and set with sample values.

### Job Scripts

All the scripts we used to run our experiments with different models and also pre computing the similarities between cases are included in the `job_scripts` directory. 


### Crawler

Part of our study being concentrated on extracting case representations from arguments such as their `explanations`, `structure`, `goals`, and their `counterarguments`, we used a crawler to extract these representations from the arguments. The crawler is in `crawler` directory. As it uses the OpenAI api and needs the api key, you have to include all the keys in the `config.py` file as a dictionary with the keys being indices starting from 0 and values being the keys.


Please use the following citation if you use this code in your work:

```
@article{sourati2023case,
  title={Case-based reasoning with language models for classification of logical fallacies},
  author={Sourati, Zhivar and Ilievski, Filip and Sandlin, H{\^o}ng-{\^A}n and Mermoud, Alain},
  journal={arXiv preprint arXiv:2301.11879},
  year={2023}
}
```