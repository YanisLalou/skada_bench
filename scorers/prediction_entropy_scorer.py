from skada.metrics import PredictionEntropyScorer
from base_bench_scorer import BaseBenchScorer

class PredictionEntropyBenchScorer(BaseBenchScorer):

    def get_scorer(self):
        return PredictionEntropyScorer()
    
    def __str__(self):
        return 'PredictionEntropyScorer'
