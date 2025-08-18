# File: training/distributed_trainer.py (NEW)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedMTLTrainer(MTLTrainer):
    """Extended trainer with distributed training support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize distributed training
        if torch.cuda.device_count() > 1:
            self.setup_distributed()
    
    def setup_distributed(self):
        """Setup distributed training."""
        dist.init_process_group(backend='nccl')
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(self.local_rank)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )
        
        # Use DistributedSampler for data loading
        from torch.utils.data.distributed import DistributedSampler
        self.train_sampler = DistributedSampler(self.train_dataset)