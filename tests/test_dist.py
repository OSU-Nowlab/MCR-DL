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
import pytest
import torch

import mcr_dl.utils as utils
from .common import DistributedTest, DistributedFixture, get_master_port, set_accelerator_visible
from .simple_model import SimpleModel
from mcr_dl.cuda_accelerator import get_accelerator
from mcr_dl.comm import mpi_discovery
from mcr_dl.utils import set_mpi_dist_environemnt

def init_mcr_dl_dist(args_backend):
    global dist
    import mcr_dl
    import mcr_dl as dist
    mcr_dl.init_distributed(dist_backend=args_backend, use_mcr_dl=True)
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)

def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist
    if backend == 'nccl':
        mpi_discovery()
    elif backend == 'mpi':
        set_mpi_dist_environemnt()
    dist.init_process_group(backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    #get_accelerator().set_device(local_rank)


@pytest.fixture(scope="session", autouse=True)
def setup_session(pytestconfig):
    # Initialization for test session
    set_accelerator_visible()
    _backend = pytestconfig.getoption("backend")
    _dist = pytestconfig.getoption("dist")
    if _dist == 'mcr_dl':
        init_mcr_dl_dist(_backend)
    elif _dist == 'torch':
        init_torch_distributed(_backend)
    else:
        print(f"distributed framework {_dist} not supported")
        exit(0)

# class TestInit(DistributedTest):
#     world_size = 3

#     def test(self):
#         assert dist.is_initialized()
#         assert dist.get_world_size() == 3
#         assert dist.get_rank() < 3


# # Demonstration of pytest's parameterization and fixtures
# @pytest.fixture(params=["hello"])
# def greeting(request):
#     return request.param


# @pytest.mark.parametrize("number,color", [(1138, "purple")])
# class TestDistArgs(DistributedTest):
#     world_size = 2
#     """ Classes that use DistributedTest class must define a test* method """

#     @pytest.mark.parametrize("shape", ["icosahedron"])
#     def test(self, number, color, shape, greeting):
#         """Ensure that we can parse args to DistributedTest methods. """
#         assert dist.get_world_size() == 2
#         assert number == 1138
#         assert color == "purple"
#         assert shape == "icosahedron"
#         assert greeting == "hello"


# # Demonstration of distributed tests grouped in single class
# @pytest.mark.parametrize("number", [1138])
# class TestGroupedDistTest(DistributedTest):
#     world_size = 2

#     def test_one(self, number):
#         assert dist.get_world_size() == 2
#         assert number == 1138

#     def test_two(self, number, color="purple"):
#         assert dist.get_world_size() == 2
#         assert number == 1138
#         assert color == "purple"


# Demonstration of world_size override
# class TestWorldSizeOverrideDistTest(DistributedTest):
#     world_size = 2

#     def test_world_size_2(self):
#         assert dist.get_world_size() == 2

#     @pytest.mark.world_size(1)
#     def test_world_size_1(self):
#         assert dist.get_world_size() == 1


# # Demonstration of the DistributedFixture class
# @pytest.fixture(params=[2, 4])
# def val1(request):
#     return request.param


# @pytest.fixture(params=[16, 32])
# def val2(request):
#     return request.param


class distributed_fixture(DistributedFixture):
    world_size = 2

    def run(self, class_tmpdir, val1, val2):
        assert int(os.environ["WORLD_SIZE"]) == self.world_size
        local_rank = os.environ["LOCAL_RANK"]
        file_path = os.path.join(class_tmpdir, f"checkpoint-{local_rank}.pt")
        with open(file_path, "w") as f:
            f.write(f"{local_rank},{val1},{val2}")


# class TestDistributedFixture(DistributedTest):
#     world_size = 1

#     def test(self, distributed_fixture, class_tmpdir, val1, val2):
#         for rank in range(2):
#             file_path = os.path.join(class_tmpdir, f"checkpoint-{rank}.pt")
#             with open(file_path, "r") as f:
#                 chkpt = f.read()
#             assert chkpt == f"{rank},{val1},{val2}"
#         assert int(os.environ["WORLD_SIZE"]) == 1


class TestDistAllReduce(DistributedTest):
    device_count = get_accelerator().device_count()
    if device_count >= 4:
        world_size = [1, 2, 4]
    elif device_count >= 2:
        world_size = [1, 2]
    else:
        world_size = [1]

    def test(self):
        x = torch.ones(1, 3).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
        sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
        result = torch.ones(1, 3).to(get_accelerator().device_name()) * sum_of_ranks
        dist.all_reduce(x)
        assert torch.all(x == result)


# class TestDistInferenceAllReduce(DistributedTest):
#     world_size = 4

#     def test(self):
#         x = torch.ones(1, 3).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
#         sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
#         result = torch.ones(1, 3).to(get_accelerator().device_name()) * sum_of_ranks
#         dist.inference_all_reduce(x)
#         assert torch.all(x == result)


# @pytest.mark.parametrize("dist_init_required", [True, False, None])
# class TestDistInit(DistributedTest):
#     init_distributed = False

#     def test_already_init(self, dist_init_required):
#         torch.distributed.init_process_group(get_accelerator().communication_backend_name())
#         mcr_dl.init_distributed(get_accelerator().communication_backend_name(),
#                                    dist_init_required=dist_init_required)

#     def test_no_init(self, dist_init_required):
#         if dist_init_required or dist_init_required is None:
#             mcr_dl.init_distributed(get_accelerator().communication_backend_name(),
#                                        dist_init_required=dist_init_required)
#         else:
#             # torch.dist is not done and for some reason the user says they don't want it done
#             with pytest.raises(Exception):
#                 mcr_dl.init_distributed(get_accelerator().communication_backend_name(),
#                                            dist_init_required=dist_init_required)


# class TestDistInitNoEnv(DistributedTest):
#     world_size = 1
#     init_distributed = False
#     set_dist_env = False

#     def test(self):
#         torch.distributed.init_process_group(backend=get_accelerator().communication_backend_name(),
#                                              init_method=f"tcp://127.0.0.1:{get_master_port()}",
#                                              world_size=1,
#                                              rank=0)
#         assert torch.distributed.is_initialized()
#         mcr_dl.init_distributed(get_accelerator().communication_backend_name(), auto_mpi_discovery=True)


# @pytest.mark.parametrize("dist_init_required", [True, False])
# class TestDistInitWithModel(DistributedTest):
#     init_distributed = False

#     def test_already_init(self, dist_init_required):
#         torch.distributed.init_process_group(get_accelerator().communication_backend_name())
#         model = SimpleModel(4)
#         config_dict = {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {}}}
#         engine, *_ = mcr_dl.initialize(model=model,
#                                           config=config_dict,
#                                           model_parameters=model.parameters(),
#                                           dist_init_required=dist_init_required)

#     def test_no_init(self, dist_init_required):
#         model = SimpleModel(4)
#         config_dict = {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {}}}
#         if dist_init_required:
#             engine, *_ = mcr_dl.initialize(model=model,
#                                               config=config_dict,
#                                               model_parameters=model.parameters(),
#                                               dist_init_required=dist_init_required)
#         else:
#             # torch.dist is not done and for some reason the user says they don't want it done
#             with pytest.raises(Exception):
#                 engine, *_ = mcr_dl.initialize(model=model,
#                                                   config=config_dict,
#                                                   model_parameters=model.parameters(),
#                                                   dist_init_required=dist_init_required)

# """