from base_bench_dataset import BaseBenchDataset

from sklearn.linear_model import LogisticRegression

from skada.datasets import make_shifted_datasets
from sklearn.preprocessing import FunctionTransformer

class ShiftedTestDataset(BaseBenchDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded_dataset = make_shifted_datasets(
            n_samples_source=300,
            n_samples_target=200,
            shift="concept_drift",
            label="binary",
            noise=0.4,
            random_state=0,
            return_dataset=True
        )

    def pack_train(self):
        X, y, sample_domain = self.loaded_dataset.pack_train(as_sources=['s'], as_targets=['t'], return_X_y=True)

        return X, y, sample_domain
    
    def pack_test(self):
        X, y, sample_domain = self.loaded_dataset.pack(as_sources=['s'], as_targets=['t'], return_X_y=True, train=False)

        return X, y, sample_domain


    def get_prepocessor(self):
        # Do nothing
        return FunctionTransformer()
    
    def get_classifier(self):
        return LogisticRegression()
    
    def __str__(self) -> str:
        return f'ShiftedTestDataset'

    bibliography = None
