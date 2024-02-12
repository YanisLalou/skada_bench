from abc import abstractmethod

class BaseBenchEstimator():
    """
    Base class for all estimators
    """

    @abstractmethod
    def get_base_estimator(self):
        pass
