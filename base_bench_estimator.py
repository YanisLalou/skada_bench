from abc import abstractmethod
from base_bench_class import BaseBenchClass

class BaseBenchEstimator(BaseBenchClass):
    """
    Base class for all estimators
    """

    @abstractmethod
    def get_base_estimator(self):
        pass

    @abstractmethod
    def get_citation(self):
        pass