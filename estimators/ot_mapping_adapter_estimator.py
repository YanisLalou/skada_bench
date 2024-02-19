from skada import OTMappingAdapter
from sklearn.base import BaseEstimator

from base_bench_estimator import BaseBenchEstimator

class OTMappingAdapterEstimator(BaseBenchEstimator):
    parameters = {'otmappingadapterestimator__metric': ['sqeuclidean'],
                  'otmappingadapterestimator__norm': [None, 'median', 'max'],
                  'otmappingadapterestimator__max_iter': [100_000]}

    def __init__(
        self,
        metric="sqeuclidean",
        norm=None,
        max_iter=100_000,
    ):
        self.metric = metric
        self.norm = norm
        self.max_iter = max_iter
        self.base_estimator = OTMappingAdapter(
            metric=self.metric,
            norm=self.norm,
            max_iter=self.max_iter
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