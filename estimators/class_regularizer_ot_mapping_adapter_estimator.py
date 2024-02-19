from skada import ClassRegularizerOTMappingAdapter
from sklearn.base import BaseEstimator

from base_bench_estimator import BaseBenchEstimator

class ClassRegularizerOTMappingAdapterEstimator(BaseBenchEstimator):
    parameters = {'classregularizerotmappingadapterestimator__reg_e': [1., 10.],
                    'classregularizerotmappingadapterestimator__reg_cl': [0.01, 0.1],
                    'classregularizerotmappingadapterestimator__norm': ['lpl1', 'l1l2'],
                    'classregularizerotmappingadapterestimator__metric': ['sqeuclidean'],
                    'classregularizerotmappingadapterestimator__max_iter': [10],
                    'classregularizerotmappingadapterestimator__max_inner_iter': [200],
                    'classregularizerotmappingadapterestimator__tol': [10e-9]}

    def __init__(
        self,
        reg_e=1.,
        reg_cl=0.1,
        norm="lpl1",
        metric="sqeuclidean",
        max_iter=10,
        max_inner_iter=200,
        tol=10e-9,
    ):
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.norm = norm
        self.metric = metric
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol

        self.base_estimator = ClassRegularizerOTMappingAdapter(
            reg_e=self.reg_e,
            reg_cl=self.reg_cl,
            norm=self.norm,
            metric=self.metric,
            max_iter=self.max_iter,
            max_inner_iter=self.max_inner_iter,
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