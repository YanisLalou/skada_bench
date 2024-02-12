from sklearn.base import BaseEstimator

from skada import CORALAdapter, Shared
from skada import make_da_pipeline

from sklearn.linear_model import LogisticRegression
from base_bench_estimator import BaseBenchEstimator

class CORALAdapterEstimator(BaseBenchEstimator):
    parameters = {'coraladapter__reg': ['auto', 0, 1]}

    def __init__(self, reg='auto', **kwargs):
        self.reg = reg
        self.base_estimator = CORALAdapter(reg = self.reg)
        self.base_estimator.set_params(**kwargs)
        
    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator
    
    def __str__(self) -> str:
        return f'CORALAdapterEstimator'
