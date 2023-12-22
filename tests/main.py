# Copyright 2023, The Ohio State University. All rights reserved.
# The MVAPICH software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse
import torch

from mcr_dl.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from common import set_accelerator_visible
from mcr_dl.cuda_accelerator import get_accelerator
from mcr_dl.comm import mpi_discovery
from mcr_dl.utils import set_mpi_dist_environemnt

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=['mpi', 'nccl'], help = "Backend")
parser.add_argument("--dist", choices=['mcr_dl', 'torch'], help = "torch.distributed or mcr-dl for distributed")
args = parser.parse_args()

def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist
    if backend == 'nccl':
        mpi_discovery()
    elif backend == 'mpi':
        set_mpi_dist_environemnt()
    dist.init_process_group(backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)


def init_mcr_dl_comm(backend):
    global dist
    import mcr_dl
    import mcr_dl as dist
    mcr_dl.init_distributed(dist_backend=backend, use_mcr_dl=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    #get_accelerator().set_device(local_rank)


def init_processes(args_dist, args_backend):
    print(f'Comm : {args_dist}  Backend : {args_backend}')
    if args_dist == 'mcr_dl':
        init_mcr_dl_comm(args_backend)
    elif args_dist == 'torch':
        init_torch_distributed(args_backend)
    else:
        print(f"distributed framework {args_dist} not supported")
        exit(0)

def all_reduce():
    x = torch.ones(1, 3).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
    sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
    result = torch.ones(1, 3).to(get_accelerator().device_name()) * sum_of_ranks
    dist.all_reduce(x)
    assert torch.all(x == result)

def all_reduce_benchmark():
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2, 30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2, 30)]
    rank = dist.get_rank()
    itr = 0

    for power in range(2, 30):
        bytes = 2 ** power
        count = int(bytes/4)
        start_events[itr].record()

        x = torch.ones(1, count).to(get_accelerator().device_name()) * (rank + 1)
        sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
        result = torch.ones(1, count).to(get_accelerator().device_name()) * sum_of_ranks
        dist.all_reduce(x)

        end_events[itr].record()
        itr += 1
        assert torch.all(x == result)

    torch.cuda.synchronize()

    if rank == 0:
        print(f"Rank, Operation, Message Size(bytes), Time required(ms)")
        t = time.time()
        for i in range(len(start_events)):
            t = start_events[i].elapsed_time(end_events[i])
            print(f"{rank}, Allreduce, {2 ** (i + 2)}, {t:.4f}")

if __name__ == "__main__":
    set_accelerator_visible()
    init_processes(args_dist=args.dist, args_backend=args.backend)
    # all_reduce()
    all_reduce_benchmark()


print("finished......")