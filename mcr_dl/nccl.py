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

from mcr_dl.ops.comm.nccl import build_nccl_op
from mcr_dl.ops.comm.mpi import build_mpi_op
from mcr_dl.utils import logger

from .utils import *
from .backend import *
from .comm import ReduceOp

cupy = None


class NCCLBackend(Backend):
    def __init__(self, name='nccl', rank=0, size=1, mpu=None):
        super(NCCLBackend, self).__init__()
        # has_allgather_base is needed for torch. Included here for compatibility with ds comms
        self.has_allgather_base = True
        self.name = 'nccl'
        self.nccl_comm_op = build_nccl_op()
        self.mpi_comm_op = build_mpi_op()
        #self.reduce_op = build_op().ReduceOp
        self.mpi_comm_op.initialize()
        #self.rank = get_local_rank_from_launcher()
        #self.size = get_world_size_from_launcher()
        self.rank = self.mpi_comm_op.get_rank(0)
        self.size = self.mpi_comm_op.get_world_size(0)
        self.enable_onebit = False
        self.init_process_group()

        if mpu is not None:
            # handle the mpu case later
            #self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
            self.mpu = mpu
            #self.world_group = self.mpu.get_data_parallel_group()

    def has_all_gather_into_tensor(self):
        return self.all_gather_base is not None

    def init_process_group(self):
        logger.info(
            f"Initializing MCR-DL's {self.name} Communication Backend with rank = {self.rank} and size = {self.size}"
        )
        if self.size <= 0:
            # Do not initialize torch distributed but only yourself
            self.initialized = True
            # Future functionality to support ds.initialize() on a single GPU
            self.single_gpu_mode = True
        else:
            self.nccl_comm_op.initialize(self.rank, self.size)
            self.initialized = True
            self.single_gpu_mode = False

    def destroy_process_group(self, group=None):
        pass

    def new_group(self, ranks):
        return self.nccl_comm_op.new_group(ranks)
        # TODO: Change this to use comm_op.new_group when the impl. is ready.
        #if not torch.distributed.is_initialized():
        #    from mcr_dl.torch_backend import TorchBackend
        #    d = TorchBackend(rank=self.rank, size=self.size)
        #logger.info(f"new group called with {ranks}")
        #return torch.distributed.new_group(ranks)

    def test_set(self):
        self.nccl_comm_op.test_set()

    def get_rank(self, group=None):
        return self.mpi_comm_op.get_rank(0)

    def get_world_size(self, group=None):
        return self.mpi_comm_op.get_world_size(0)

    def is_initialized(self):
        return self.initialized

    def get_world_group(self):
        return self.nccl_comm_op.get_world_group()

    def barrier(self, group=None, async_op=False):
        self.mpi_comm_op.barrier()

    def broadcast(self, tensor, src, op=ReduceOp.SUM, group=None, async_op=False, block=False):
        # TODO: Fix calls to op. Fix op to support groups and async
        self.nccl_comm_op.broadcast(tensor,
                                    src,
                                    block,
                                    group,
                                    async_op)  #, group=group, async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0, block=False, async_op=False):
        self.nccl_comm_op.send(tensor, dst, tag, block, group, async_op)

    def recv(self, tensor, src=None, group=None, tag=0, block=False, async_op=False):
        self.nccl_comm_op.recv(tensor, src, tag, block, group, async_op)

    def all_reduce(self,
                   tensor,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   block=False):
        self.nccl_comm_op.all_reduce(tensor, op, block, group, async_op)

    def reduce(self,
               tensor,
               dst,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               block=False):
        self.nccl_comm_op.reduce(tensor, dst, op, block, group, async_op)

    def reduce_scatter(self,
                       tensor,
                       op=ReduceOp.SUM,
                       group=None,
                       async_op=False,
                       block=False):
        self.nccl_comm_op.reduce_scatter(tensor, op, block, group, async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False, block=False):
        self.nccl_comm_op.all_gather([tensor_list], [tensor], block, group, async_op)

    def all_gather_base(self,
                        output_tensor,
                        input_tensor,
                        group=None,
                        async_op=False,
                        block=False,
                        comm_id=0):
        self.nccl_comm_op.all_gather_base(output_tensor,
                                          input_tensor,
                                          block,
                                          group,
                                          async_op)

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False, block=False):
        self.nccl_comm_op.all_gather_base(output_tensor,
                                          input_tensor,
                                          block,
                                          group,
                                          async_op)
    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          op=ReduceOp.SUM,
                          group=None,
                          async_op=False,
                          block=False):
        self.nccl_comm_op.all_to_all_single(output, input, block, group, async_op)

    def all_to_all(self,
                   output_tensor_list,
                   input_tensor_list,
                   group=None,
                   async_op=False,
                   block=False):
        self.nccl_comm_op.all_to_all(output_tensor_list, input_tensor_list, block, group, async_op)

    def synchronize(self):
        self.nccl_comm_op.synchronize()

    def create_comm_group(self, comm_ranks, rank, comm_id, color):
        self.nccl_comm_op.create_comm_group(comm_ranks, rank, comm_id, color)