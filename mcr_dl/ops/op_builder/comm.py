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

from .builder import CUDAOpBuilder
from .config import ConfigPath

config = ConfigPath()
OMPI = config.mpi_path
MPI_INCL = config.mpi_include
MPI_LIB = '-L' + OMPI + '/lib -lmpi'

CUDA = config.cuda_path
CUDA_INCL = config.cuda_include
CUDA_LIB = '-L' + CUDA + '/lib64 -lm -lcuda -lcudart'

NCCL = config.nccl_path
NCCL_INCL = config.nccl_include

class NCCLCommBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_NCCL_COMM"
    NAME = "mcr_dl_nccl_comm"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'mcr_dl.ops.comm.{self.NAME}_op'

    def sources(self):
        return ['csrc/comm/nccl.cpp']

    def include_paths(self):
        includes = ['csrc/includes']
        return includes + [MPI_INCL, CUDA_INCL, NCCL_INCL]

    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        return [MPI_LIB, CUDA_LIB]


class MPICommBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_MPI_COMM"
    NAME = "mcr_dl_mpi_comm"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'mcr_dl.ops.comm.{self.NAME}_op'

    def sources(self):
        return ['csrc/comm/mpi.cpp']

    def include_paths(self):
        includes = ['csrc/includes']
        return includes + [MPI_INCL, CUDA_INCL, NCCL_INCL]

    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        return [MPI_LIB, CUDA_LIB]