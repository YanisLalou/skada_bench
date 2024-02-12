from enum import Enum

from base_bench_dataset import BaseBenchDataset
from base_bench_estimator import BaseBenchEstimator
from base_bench_scorer import BaseBenchScorer

class BaseBenchClasses(Enum):
    dataset = BaseBenchDataset
    estimator = BaseBenchEstimator
    scorer = BaseBenchScorer
