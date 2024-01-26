from skada import SubspaceAlignmentAdapter

class SubspaceAlignmentAdapterEstimator(SubspaceAlignmentAdapter):
    def __init__(
        self,
        n_components=None,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            random_state=random_state,
        )

    def adapt(self, X, y=None, sample_domain=None, **kwargs):
        return super().adapt(X, y=y, sample_domain=sample_domain, **kwargs)
    
    def fit(self, X, y=None, sample_domain=None, **kwargs):
        return super().fit(X, y=y, sample_domain=sample_domain, **kwargs)

