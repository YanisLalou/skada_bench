from base_bench_dataset import BaseBenchDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from skada.datasets import fetch_office31_surf_all


class Office31SurfPca2Dataset(BaseBenchDataset):

    def __init__(self) -> None:
        super().__init__()
        self.loaded_dataset = fetch_office31_surf_all()

    def pack_train(self):
        X, y, sample_domain = self.loaded_dataset.pack_train(as_sources=['amazon', 'webcam'], as_targets=['dslr'], return_X_y=True)

        return X, y, sample_domain
    
    def pack_test(self):
        X, y, sample_domain = self.loaded_dataset.pack(as_sources=['amazon', 'webcam'], as_targets=['dslr'], return_X_y=True, train=False)

        return X, y, sample_domain

    def get_prepocessor(self):
        pca = PCA(n_components=2)
        return pca
    
    def get_classifier(self):
        return LogisticRegression()
    
    def __str__(self) -> str:
        return f'Office31SurfPca2Dataset'

