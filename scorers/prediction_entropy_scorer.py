from skada.metrics import PredictionEntropyScorer

class PredictionEntropyScorer(PredictionEntropyScorer):
    def __init__(self, greater_is_better=False):
        super().__init__(greater_is_better=greater_is_better)
        
    def _score(self, estimator, X, y, sample_domain=None, **params):
        return super()._score(estimator, X, y, sample_domain=sample_domain, **params)
