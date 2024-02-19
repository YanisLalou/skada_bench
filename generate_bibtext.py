# This script is used to generate the bibtext file
# containing the references of the datasets, estimators and scorers.
import argparse

from utils import fetch_all_classes, generate_bibtext
from base_bench_classes import BaseBenchClasses

import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasets", help = "Directory containing the datasets", type = str, default = "datasets")
parser.add_argument("-e", "--estimators", help = "Directory containing the estimators", type = str, default = "estimators")
parser.add_argument("-s", "--scorers", help = "Directory containing the scorers", type = str, default = "scorers")


def generate_bibtext_all(args):
    # Fetch the datasets
    dataset_classes = fetch_all_classes(args.datasets, BaseBenchClasses.dataset.value)

    # Fetch the estimators
    estimator_classes = fetch_all_classes(args.estimators, BaseBenchClasses.estimator.value)

    # Fetch the scoring functions
    scorer_classes = fetch_all_classes(args.scorers, BaseBenchClasses.scorer.value)

    generate_bibtext(dataset_classes + estimator_classes + scorer_classes)


if __name__ == "__main__":
    args = parser.parse_args()

    generate_bibtext_all(args)
