from skada import EntropicOTMappingAdapter

class EntropicOTMappingAdapterEstimator(EntropicOTMappingAdapter):
    def __init__(self):
        super().__init__()

    def adapt(self, X, y=None, sample_domain=None, **kwargs):
        return super().adapt(X, y=y, sample_domain=sample_domain, **kwargs)
    
    def fit(self, X, y=None, sample_domain=None, **kwargs):
        return super().fit(X, y=y, sample_domain=sample_domain, **kwargs)

    def __str__(self):
        return 'EntropicOTMappingAdapter'