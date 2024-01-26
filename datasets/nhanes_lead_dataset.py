from typing import Iterable, List, Mapping, Optional, Tuple, Union, Dict
import numpy as np
from skada.datasets import DomainAwareDataset, fetch_domain_aware_nhanes_lead
from skada.datasets._base import PackedDatasetType, DomainDataType

class NhanesLeadDataset(DomainAwareDataset):
    def __init__(
        self,
        # xxx(okachaiev): not sure if dictionary is a good format :thinking:
        domains: Union[List[DomainDataType], Dict[str, DomainDataType], None] = None
    ):
        super().__init__(domains)
        loaded_dataset = fetch_domain_aware_nhanes_lead()
        self.domains_ = loaded_dataset.domains_
        self.domain_names_ = loaded_dataset.domain_names_

    def add_domain(
        self,
        X,
        y=None,
        domain_name: Optional[str] = None
    ) -> 'DomainAwareDataset':
        return super().add_domain(X, y, domain_name)
    
    def merge(
        self,
        dataset: 'DomainAwareDataset',
        names_mapping: Optional[Mapping] = None
    ) -> 'DomainAwareDataset':
        return super().merge(dataset, names_mapping)
    
    def get_domain(self, domain_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return super().get_domain(domain_name)
    
    def select_domain(
        self,
        sample_domain: np.ndarray,
        domains: Union[str, Iterable[str]]
    ) -> np.ndarray:
        return super().select_domain(sample_domain, domains)
    
    def pack(
        self,
        as_sources: List[str] = None,
        as_targets: List[str] = None,
        return_X_y: bool = True,
        train: bool = False,
        mask: Union[None, int, float] = None,
    ) -> PackedDatasetType:
        return super().pack(as_sources, as_targets, return_X_y, train, mask)
    
    def pack_train(
        self,
        as_sources: List[str],
        as_targets: List[str],
        return_X_y: bool = True,
        mask: Union[None, int, float] = None,
    ) -> PackedDatasetType:
        return super().pack_train(as_sources, as_targets, return_X_y, mask)
    
    def pack_test(
        self,
        as_targets: List[str],
        return_X_y: bool = True,
    ) -> PackedDatasetType:
        return super().pack_test(as_targets, return_X_y)
    
    def pack_lodo(self, return_X_y: bool = True) -> PackedDatasetType:
        return super().pack_lodo(return_X_y)