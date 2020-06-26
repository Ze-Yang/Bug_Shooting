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
        self.net2 = nn.Linear(10, 5, bias=False)

    def forward(self, x):
        x1 = self.relu(self.net1(x))  # [20, 10]
        x1_list = [torch.empty_like(x1, device='cuda') for _ in range(dist.get_world_size())]
        dist.all_gather(x1_list, x1)
        y = torch.cat(x1_list, dim=0).mean(0, keepdim=True).expand(5, -1)  # [5, 10]
        weight = 0.9 * self.net2.weight + 0.1 * y
        out = x1.mm(weight.t())
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

    optimizer.zero_grad()
    inputs = torch.randn((20, 10), device='cuda')
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5).to('cuda')
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
