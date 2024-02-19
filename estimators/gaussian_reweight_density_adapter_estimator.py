from skada import GaussianReweightDensityAdapter
from sklearn.base import BaseEstimator

from base_bench_estimator import BaseBenchEstimator

class GaussianReweightDensityAdapterEstimator(BaseBenchEstimator):
    parameters = {'gaussianreweightdensityadapter__reg': ['auto']}

    def __init__(self, reg='auto'):
        self.reg = reg
        self.base_estimator = GaussianReweightDensityAdapter(reg=self.reg)

    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator

    bibliography = (
        """@article{SHIMODAIRA2000227,
        title = {Improving predictive inference under covariate shift by weighting the log-likelihood function},
        journal = {Journal of Statistical Planning and Inference},
        volume = {90},
        number = {2},
        pages = {227-244},
        year = {2000},
        issn = {0378-3758},
        doi = {https://doi.org/10.1016/S0378-3758(00)00115-4},
        url = {https://www.sciencedirect.com/science/article/pii/S0378375800001154},
        author = {Hidetoshi Shimodaira},
        keywords = {Akaike information criterion, Design of experiments, Importance sampling, Kullback–Leibler divergence, Misspecification, Sample surveys, Weighted least squares},
        abstract = {A class of predictive densities is derived by weighting the observed samples in maximizing the log-likelihood function. This approach is effective in cases such as sample surveys or design of experiments, where the observed covariate follows a different distribution than that in the whole population. Under misspecification of the parametric model, the optimal choice of the weight function is asymptotically shown to be the ratio of the density function of the covariate in the population to that in the observations. This is the pseudo-maximum likelihood estimation of sample surveys. The optimality is defined by the expected Kullback–Leibler loss, and the optimal weight is obtained by considering the importance sampling identity. Under correct specification of the model, however, the ordinary maximum likelihood estimate (i.e. the uniform weight) is shown to be optimal asymptotically. For moderate sample size, the situation is in between the two extreme cases, and the weight function is selected by minimizing a variant of the information criterion derived as an estimate of the expected loss. The method is also applied to a weighted version of the Bayesian predictive density. Numerical examples as well as Monte-Carlo simulations are shown for polynomial regression. A connection with the robust parametric estimation is discussed.}
        }"""
    )