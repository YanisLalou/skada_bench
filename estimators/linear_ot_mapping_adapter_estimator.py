from sklearn.base import BaseEstimator
from skada import LinearOTMappingAdapter

from base_bench_estimator import BaseBenchEstimator

class LinearOTMappingAdapterEstimator(BaseBenchEstimator):
    parameters = {'linearotmappingadapter__reg': [1e-08, 1e-06],
                  'linearotmappingadapter__bias': [True, False]}

    def __init__(self, reg=1e-08, bias=True, **kwargs):
        self.reg = reg
        self.bias = bias
        self.base_estimator = LinearOTMappingAdapter(reg = self.reg, bias = self.bias)
        self.base_estimator.set_params(**kwargs)
        
    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator
    
    def __str__(self) -> str:
        return f'LinearOTMappingAdapterEstimator'

    bibliography = (
        """@misc{courty2016optimal,
            title={Optimal Transport for Domain Adaptation}, 
            author={Nicolas Courty and Rémi Flamary and Devis Tuia and Alain Rakotomamonjy},
            year={2016},
            eprint={1507.00504},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }"""
    )