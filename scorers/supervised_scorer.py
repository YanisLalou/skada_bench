from skada.metrics import SupervisedScorer
from base_bench_scorer import BaseBenchScorer

class SupervisedBenchScorer(BaseBenchScorer):

    def get_scorer(self):
        return SupervisedScorer()
    
    def __str__(self):
        return 'SupervisedScorer'

    bibliography = None