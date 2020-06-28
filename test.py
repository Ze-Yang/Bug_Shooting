import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.multiprocessing as mp
import comm


parser = argparse.ArgumentParser(description='Distributed Data Parallel')
parser.add_argument('--world-size', type=int, default=2,
                    help='Number of GPU(s).')


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.stem = nn.Linear(10, 10)
        self.branch1 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10))
        self.branch2 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10))

    def forward(self, x):
        x1 = F.relu(self.stem(x))  # [20, 10]
        branch1 = self.branch1(x1[:10])
        branch2 = self.branch2(x1[10:])
        branch1_list = [torch.empty_like(branch1, device='cuda') for _ in range(dist.get_world_size())]
        dist.all_gather(branch1_list, branch1)
        # branch1_list = comm.all_gather(branch1)
        pred_weight = torch.cat(branch1_list, dim=0).mean(0, keepdim=True).expand(5, -1)  # [5, 10]
        out = branch2.mm(pred_weight.t())
        return out


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to('cuda')
    ddp_model = DistributedDataParallel(model, device_ids=[dist.get_rank()], broadcast_buffers=False)
    ddp_model.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(5):
        optimizer.zero_grad()
        inputs = torch.randn((20, 10), device='cuda')
        outputs = ddp_model(inputs)
        labels = torch.randn_like(outputs).to('cuda')
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    run_demo(demo_basic, args.world_size)
