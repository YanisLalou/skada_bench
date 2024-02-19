from skada import SubspaceAlignmentAdapter
from sklearn.base import BaseEstimator

from base_bench_estimator import BaseBenchEstimator

class SubspaceAlignmentAdapterEstimator(BaseBenchEstimator):
    parameters = {'subspacealignmentadapter__n_components': [2, 4, 6]}

    def __init__(
        self,
        n_components=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.base_estimator = SubspaceAlignmentAdapter(
            n_components=self.n_components,
            random_state=self.random_state
        )

    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator

    bibliography = (
        """@INPROCEEDINGS{6751479,
        author={Fernando, Basura and Habrard, Amaury and Sebban, Marc and Tuytelaars, Tinne},
        booktitle={2013 IEEE International Conference on Computer Vision}, 
        title={Unsupervised Visual Domain Adaptation Using Subspace Alignment}, 
        year={2013},
        volume={},
        number={},
        pages={2960-2967},
        keywords={Vectors;Context;Manifolds;Principal component analysis;Support vector machines;Covariance matrices;Eigenvalues and eigenfunctions;domain adaptation;subspace alignment;object recognition},
        doi={10.1109/ICCV.2013.368}}
        """
    )