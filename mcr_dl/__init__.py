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

from .utils import *
from .comm import *

global __dist_engine
global __dist_backend

def init_torch_distributed(backend):
    import torch.distributed as dist
    if backend == 'nccl':
        mpi_discovery()
    elif backend == 'mpi':
        set_mpi_dist_environemnt()
    dist.init_process_group(backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)

def init_mcr_dl_comm(backend):
    import mcr_dl
    mcr_dl.init_distributed(dist_backend=backend, use_mcr_dl=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    #get_accelerator().set_device(local_rank)

def init_processes(dist_engine, dist_backend):
    print(f'Comm : {dist_engine}  Backend : {dist_backend}')

    global __dist_engine
    global __dist_backend
    __dist_engine = dist_engine
    __dist_backend = dist_backend

    if dist_engine == 'mcr_dl':
        init_mcr_dl_comm(dist_backend)
    elif dist_engine == 'torch':
        init_torch_distributed(dist_backend)
    else:
        print(f"distributed framework {dist_engine} not supported")
        exit(0)

def get_distributed_engine():
    global __dist_engine
    if __dist_engine == 'torch':
        return torch.distributed
    elif __dist_engine == 'mcr_dl':
        import mcr_dl
        return mcr_dl