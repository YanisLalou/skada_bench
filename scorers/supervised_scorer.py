from skada.metrics import SupervisedScorer

class SupervisedScorer(SupervisedScorer):
    def __init__(self, greater_is_better=False):
        super().__init__(greater_is_better=greater_is_better)
        
    def _score(self, estimator, X, y, sample_domain=None, **params):
        return super()._score(estimator, X, y, sample_domain=sample_domain, **params)

    def __str__(self):
        return 'SupervisedScorer'