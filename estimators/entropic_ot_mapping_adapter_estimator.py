from skada import EntropicOTMappingAdapter
from sklearn.base import BaseEstimator

from base_bench_estimator import BaseBenchEstimator

class EntropicOTMappingAdapterEstimator(BaseBenchEstimator):
    parameters = {'entropicotmappingadapterestimator__reg_e': [1, 10],
                  'entropicotmappingadapterestimator__metric': ['sqeuclidean'],
                  'otmappingadapterestimator__norm': [None, 'median', 'max'],
                  'otmappingadapterestimator__max_iter': [1000],
                  'otmappingadapterestimator__tol': [10e-9]}

    def __init__(
        self,
        reg_e=1.,
        metric="sqeuclidean",
        norm=None,
        max_iter=1000,
        tol=10e-9,
    ):
        self.reg_e = reg_e
        self.metric = metric
        self.norm = norm
        self.max_iter = max_iter
        self.tol = tol

        self.base_estimator = EntropicOTMappingAdapter(
            reg_e=self.reg_e,
            metric=self.metric,
            norm=self.norm,
            max_iter=self.max_iter,
            tol=self.tol
        )

    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator

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