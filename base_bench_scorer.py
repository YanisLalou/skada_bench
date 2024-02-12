from abc import abstractmethod

class BaseBenchScorer():
    """
    Base class for all scorers
    """

    @abstractmethod
    def get_scorer(self):
        pass
