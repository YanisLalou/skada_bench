from skada.metrics import SupervisedScorer
from base_bench_scorer import BaseBenchScorer
from sklearn.metrics import accuracy_score, make_scorer

class SupervisedBenchScorer(BaseBenchScorer):

    def get_scorer(self):
        return SupervisedScorer()
    
    def __str__(self):
        return 'SupervisedScorer'
