from abc import abstractmethod
from base_bench_class import BaseBenchClass

class BaseBenchScorer(BaseBenchClass):
    """
    Base class for all scorers
    """

    @abstractmethod
    def get_scorer(self):
        pass

    @abstractmethod
    def get_citation(self):
        pass