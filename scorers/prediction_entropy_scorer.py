from skada.metrics import PredictionEntropyScorer
from base_bench_scorer import BaseBenchScorer

class PredictionEntropyBenchScorer(BaseBenchScorer):

    def get_scorer(self):
        return PredictionEntropyScorer()
    
    def __str__(self):
        return 'PredictionEntropyScorer'

    bibliography = ("""
                    @article{DBLP:journals/corr/abs-1711-10288,
                    author       = {Pietro Morerio and
                                    Jacopo Cavazza and
                                    Vittorio Murino},
                    title        = {Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain
                                    Adaptation},
                    journal      = \{CoRR\},
                    volume       = {abs/1711.10288},
                    year         = {2017},
                    url          = {http://arxiv.org/abs/1711.10288},
                    eprinttype    = {arXiv},
                    eprint       = {1711.10288},
                    timestamp    = {Mon, 13 Aug 2018 16:48:30 +0200},
                    biburl       = {https://dblp.org/rec/journals/corr/abs-1711-10288.bib},
                    bibsource    = {dblp computer science bibliography, https://dblp.org}
                    }
                    """
                    )