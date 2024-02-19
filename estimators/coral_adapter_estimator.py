from sklearn.base import BaseEstimator
from skada import CORALAdapter

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

    bibliography = (
        """@article{DBLP:journals/corr/SunFS16,
            author       = {Baochen Sun and
                            Jiashi Feng and
                            Kate Saenko},
            title        = {Correlation Alignment for Unsupervised Domain Adaptation},
            journal      = {CoRR},
            volume       = {abs/1612.01939},
            year         = {2016},
            url          = {http://arxiv.org/abs/1612.01939},
            eprinttype    = {arXiv},
            eprint       = {1612.01939},
            timestamp    = {Mon, 13 Aug 2018 16:47:30 +0200},
            biburl       = {https://dblp.org/rec/journals/corr/SunFS16.bib},
            bibsource    = {dblp computer science bibliography, https://dblp.org}
            }"""
    )