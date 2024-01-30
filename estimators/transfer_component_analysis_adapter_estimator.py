from skada import TransferComponentAnalysisAdapter

class TransferComponentAnalysisAdapterEstimator(TransferComponentAnalysisAdapter):
    def __init__(
        self,
        kernel='rbf',
        n_components=None,
        mu=0.1
    ):
        super().__init__()
        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu

    def adapt(self, X, y=None, sample_domain=None, **kwargs):
        return super().adapt(X, y=y, sample_domain=sample_domain, **kwargs)
    
    def fit(self, X, y=None, sample_domain=None, **kwargs):
        return super().fit(X, y=y, sample_domain=sample_domain, **kwargs)

    def __str__(self):
        return 'TransferComponentAnalysisAdapter'