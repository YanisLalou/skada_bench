from abc import abstractmethod

class BaseBenchClass():
    """
    Base class for all bench classes
    """

    @abstractmethod
    def get_citation(self):
        pass