import argparse
import os
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm
import sys
from abc import abstractmethod
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


class Retriever:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def retrieve_similar_cases(self, case: str, num_cases: int):
        pass


class SimCSE_Retriever(Retriever):
    def __init__(self, config) -> None:
        self.similarities_dict = dict()
        base_path = os.path.join("cache", config.data_dir.replace("/", "_"))
        simcse_model_paths = [file for file in os.listdir(
            base_path) if file.startswith(f"simcse_similarities_{config.feature}")]
        simcse_model_paths = [file for file in simcse_model_paths if file.endswith(
            f"ratio_{config.ratio_of_source_used}.joblib")]
        for path in simcse_model_paths:
            self.similarities_dict.update(
                joblib.load(os.path.join(base_path, path))
            )
        print("Loaded SimCSE similarities")
        print("Number of files loaded:", len(simcse_model_paths))

    def retrieve_similar_cases(self, case: str, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1][0], reverse=True)

        return [
            {
                'similar_case': x[0],
                'similar_case_label': x[1][1]
            } for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1][0] > threshold
        ]

# TODO: change the structure to be more similar to the SimCSE_Retriever and remove the train_df and other unnecessary parameters


class SentenceTransformerRetriever(Retriever):
    def __init__(self, config) -> None:
        self.similarities_dict = dict()
        base_path = os.path.join("cache", config.data_dir.replace("/", "_"))
        simcse_model_paths = [file for file in os.listdir(
            base_path) if file.startswith(config.retriever_str.replace("/", "_"))]
        for path in simcse_model_paths:
            self.similarities_dict.update(
                joblib.load(os.path.join(base_path, path)))

    def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)

        return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]

# TODO: change the structure to be more similar to the SimCSE_Retriever and remove the train_df and other unnecessary parameters


class Empathy_Retriever(Retriever):
    def __init__(self, config) -> None:
        self.similarities_dict = dict()
        base_path = os.path.join("cache", config.data_dir.replace("/", "_"))
        empathetic_model_paths = [file for file in os.listdir(
            base_path) if file.startswith("empathy_similarities")]

        for path in empathetic_model_paths:
            self.similarities_dict.update(
                joblib.load(os.path.join(base_path, path)))

    def retrieve_similar_cases(self, case: str, train_df: pd.DataFrame, num_cases: int = 1, threshold: float = -np.inf):
        sentences_and_similarities = self.similarities_dict[case.strip()].items(
        )
        sentences_and_similarities_sorted = sorted(
            sentences_and_similarities, key=lambda x: x[1], reverse=True)

        return [(x[0], x[1], train_df[train_df["text"].str.strip() == x[0].strip()].label.tolist()) for x in sentences_and_similarities_sorted[1:num_cases + 1] if x[1] > threshold]


if __name__ == "__main__":
    pass
