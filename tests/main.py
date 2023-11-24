import os
from mcr_dl.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
import mcr_dl as dist
import torch
from common import set_accelerator_visible
from mcr_dl.cuda_accelerator import get_accelerator
from test_dist import TestDistAllReduce

def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def initialize_cuda():
    my_local_rank = env2int(
        ["MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
        0,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(my_local_rank % 4)

    torch.cuda.init()

if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
if 'MASTER_ADDR' not in os.environ:
    master_addr = '10.1.1.21'
    os.environ['MASTER_ADDR'] = master_addr
local_rank = env2int(
    ['LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID'])
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(local_rank)
rank = env2int(['RANK', 'MPI_RANKID', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'SLURM_PROCID'])
if 'RANK' not in os.environ:
    os.environ['RANK'] = str(rank)
world_size = env2int(['WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'SLURM_NPROCS'])
if 'WORLD_SIZE' not in os.environ:
    os.environ['WORLD_SIZE'] = str(world_size)

#set_accelerator_visible()
#initialize_cuda()
print("CUDA VISIBILE DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"], local_rank, world_size)
torch.distributed.init_process_group(backend='nccl')
dist.init_distributed(dist_backend="nccl")


def test():
    print("DEVICE NAME: ", get_accelerator().device_name())
    x = torch.ones(1, 3).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
    sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
    result = torch.ones(1, 3).to(get_accelerator().device_name()) * sum_of_ranks
    dist.all_reduce(x)
    assert torch.all(x == result)
# t = TestDistAllReduce()
# t.test()
test()