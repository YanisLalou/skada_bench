import os
import numpy as np
import random
#import torch

def seed_everything(seed: int=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Torch stuff
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    #     torch.use_deterministic_algorithms(True)
    
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

