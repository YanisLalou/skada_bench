from enum import Enum

from sklearn.base import BaseEstimator
from skada.datasets import DomainAwareDataset
from skada.metrics import _BaseDomainAwareScorer

class BaseClasses(Enum):
    dataset = DomainAwareDataset
    estimator = BaseEstimator
    scorer = _BaseDomainAwareScorer
