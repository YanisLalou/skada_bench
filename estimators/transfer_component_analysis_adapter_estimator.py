from sklearn.base import BaseEstimator
from skada import TransferComponentAnalysisAdapter
from base_bench_estimator import BaseBenchEstimator


class TransferComponentAnalysisAdapterEstimator(BaseBenchEstimator):
    parameters = {'transfercomponentanalysisadapter__kernel': ['rbf'],
                  'transfercomponentanalysisadapter__n_components': [2, 3],
                  'transfercomponentanalysisadapter__mu': [0.01, 0.1]}

    def __init__(
        self,
        kernel='rbf',
        n_components=None,
        mu=0.1,
        **kwargs
    ):
        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu

        self.base_estimator = TransferComponentAnalysisAdapter(
            kernel=self.kernel,
            n_components=self.n_components,
            mu=self.mu
        )

        self.base_estimator.set_params(**kwargs)
        
    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator
    
    bibliography = (
        """@ARTICLE{5640675,
        author={Pan, Sinno Jialin and Tsang, Ivor W. and Kwok, James T. and Yang, Qiang},
        journal={IEEE Transactions on Neural Networks}, 
        title={Domain Adaptation via Transfer Component Analysis}, 
        year={2011},
        volume={22},
        number={2},
        pages={199-210},
        keywords={Kernel;Optimization;Manifolds;Hilbert space;Learning systems;Feature extraction;Noise measurement;Dimensionality reduction;domain adaptation;Hilbert space embedding of distributions;transfer learning},
        doi={10.1109/TNN.2010.2091281}}
        """
    )