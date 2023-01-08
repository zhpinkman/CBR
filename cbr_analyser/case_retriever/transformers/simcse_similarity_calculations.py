import argparse
import os
import sys

import joblib
from IPython import embed
import pandas as pd
from simcse import SimCSE


def get_embeddings_simcse(model, text: str):
    return model.encode(text)


def generate_the_simcse_similarities(source_file: str, target_file: str, output_file: str, feature: str, ratio_of_source_used: float = 1.0):

    if os.path.exists(output_file):
        print(f"Output file already exists for {target_file}. Skipping...")
        return

    model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

    source_file_df = pd.read_csv(source_file).groupby("label").apply(
        lambda x: x.sample(frac=ratio_of_source_used)).reset_index(drop=True)

    train_sentences = source_file_df[feature].tolist()
    train_labels = source_file_df["label"].tolist()

    train_sentences = [x.strip() for x in train_sentences]

    all_sentences = pd.read_csv(target_file)[feature].tolist()
    all_sentences = [x.strip() for x in all_sentences]

    similarities = model.similarity(all_sentences, train_sentences)
    similarities_dict = dict()
    for sentence, row in zip(all_sentences, similarities):
        similarities_dict[sentence] = dict(zip(
            train_sentences,
            list(zip(
                row.tolist(),
                train_labels
            ))
        ))

    joblib.dump(similarities_dict, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--feature', type=str,
                        help="feature to use for calculating the similarity")
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--ratio_of_source_used', type=float, default=1.0)

    args = parser.parse_args()

    generate_the_simcse_similarities(
        args.source_file, args.target_file, args.output_file, args.feature, args.ratio_of_source_used)
