# from sklearn.base import BaseEstimator
# from skada import TransferComponentAnalysisAdapter
# from base_bench_estimator import BaseBenchEstimator


# class TransferComponentAnalysisAdapterEstimator(BaseBenchEstimator):
#     parameters = {'transfercomponentanalysisadapter__kernel': ['rbf'],
#                   'transfercomponentanalysisadapter__n_components': [2, 3],
#                   'transfercomponentanalysisadapter__mu': [0.01, 0.1]}

#     def __init__(
#         self,
#         kernel='rbf',
#         n_components=None,
#         mu=0.1,
#         **kwargs
#     ):
#         self.kernel = kernel
#         self.n_components = n_components
#         self.mu = mu

#         self.base_estimator = TransferComponentAnalysisAdapter(
#             kernel=self.kernel,
#             n_components=self.n_components,
#             mu=self.mu
#         )

#         self.base_estimator.set_params(**kwargs)
        
#     def get_base_estimator(self) -> BaseEstimator:
#         return self.base_estimator