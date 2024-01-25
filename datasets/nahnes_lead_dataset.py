from skada.datasets import fetch_domain_aware_nhanes_lead
from skada.datasets import DomainAwareDataset

class NhanesLeadDataset(DomainAwareDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.target = fetch_domain_aware_nhanes_lead()