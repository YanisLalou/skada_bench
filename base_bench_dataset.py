from abc import abstractmethod

class BaseBenchDataset():
    """
    Base class for all datasets
    """

    @abstractmethod
    def get_prepocessor(self):
        pass

    @abstractmethod
    def get_classifier(self):
        pass

    @abstractmethod
    def pack_train(self):
        pass

    @abstractmethod
    def pack_test(self):
        pass
