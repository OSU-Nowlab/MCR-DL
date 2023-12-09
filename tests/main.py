import os
from mcr_dl.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
import torch
from common import set_accelerator_visible
from mcr_dl.cuda_accelerator import get_accelerator
from mcr_dl.comm import mpi_discovery
from mcr_dl.utils import set_mpi_dist_environemnt

def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist
    if backend == 'nccl':
        mpi_discovery()
    elif backend == 'mpi':
        set_mpi_dist_environemnt()
    dist.init_process_group(backend)


def init_mcr_dl_comm(backend):
    global dist
    import mcr_dl
    import mcr_dl as dist
    mcr_dl.init_distributed(dist_backend=backend, use_mcr_dl=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)


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

if __name__ == "__main__":
    set_accelerator_visible()
    init_processes(args_dist='mcr_dl', args_backend='mpi')
    all_reduce()


print("finished......")