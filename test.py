import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.multiprocessing as mp
import detectron2.utils.comm as comm


parser = argparse.ArgumentParser(description='Distributed Data Parallel')
parser.add_argument('--world-size', type=int, default=2,
                    help='Number of GPU(s).')


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[dist.get_rank()], broadcast_buffers=False)
    ddp_model.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    inputs = torch.randn((20, 10), device=rank)
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5).to(rank)
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


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    run_demo(demo_basic, args.world_size)
