# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import mcr_dl as dist
from common import DistributedTest, DistributedFixture, get_master_port, set_accelerator_visible
from simple_model import SimpleModel
from mcr_dl.cuda_accelerator import get_accelerator
from argparse import ArgumentParser
import pytest
from mcr_dl.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
# from deepspeed.ops.op_builder import FusedAdamBuilder

# if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
#     pytest.skip("This op had not been implemented on this system.", allow_module_level=True)
def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch"
                            " utility that creates multiple distributed"
                            " processes on a single node")

    # Optional arguments for the launch helper
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                        "training")
    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="Master node (rank 0)'s address, should be either"
                        " the IP address or the hostname of node 0, for"
                        " single node multi-proc training, the"
                        " --master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communication during distributed "
                        "training")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="TBD")
    # rest from the training program
    parser.add_argument('training_script_args')
    return parser.parse_args()

def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

@pytest.fixture(scope="session", autouse=True)
def setup_session():
    # Initialization for test session
    # args = parse_args()
    # current_env = os.environ.copy()

    # # dist_world_size = 0
    # # for node_id in node_list:
    # #     gids = world_info[node_id]
    # #     dist_world_size += len(gids)
    # #     for gid in gids:
    # #         global_rank_mapping[node_id].append(curr_global_rank)
    # #         curr_global_rank += 1

    # os.environ["MASTER_ADDR"] = args.master_addr
    # os.environ["MASTER_PORT"] = str(args.master_port)
    # os.environ["WORLD_SIZE"] = str(args.world_size)
    # os.environ["RANK"] = str(args.node_rank)
    # # current_env["CROSS_SIZE"] = str(args.nnodes)
    # # current_env["LOCAL_SIZE"] = str(args.local_size)

    # discover rank/size info from env
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
    if 'MASTER_ADDR' not in os.environ:
        # try:
        #     from mpi4py import MPI
        # except ModuleNotFoundError:
        #     print(
        #         "Cannot import mpi4py and MASTER_ADDR not set. Please either install mpi4py or set the MASTER_ADDR on all ranks"
        #     )
        #     raise Exception
        # import subprocess
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # master_addr = None
        # if rank == 0:
        #     hostname_cmd = ["hostname -I"]
        #     result = subprocess.check_output(hostname_cmd, shell=True)
        #     master_addr = result.decode('utf-8').split()[0]
        # master_addr = comm.bcast(master_addr, root=0)
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

    print("\nSetting up resources for the entire test session")
    set_accelerator_visible()
    torch.distributed.init_process_group(backend='nccl')
    dist.init_distributed(dist_backend="nccl")

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