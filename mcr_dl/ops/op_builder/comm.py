import torch
from .builder import CUDAOpBuilder

MV2 = f'/opt/mvapich2/gdr/2.3.6/no-mpittool/no-openacc/cuda11.3/mofed5.4/mpirun/gnu9.3.0'
# OMPI = f'/usr/local/mpi'
# OMPI = f'/home/gulhane.2/opt/openmpi'
OMPI = f'/home/gulhane.2/mvapich2-installation/nvidia/gdr2.3.7_cuda11.6_gcc10.3.0'

MPI_INCL = OMPI + '/include'
MPI_LIB = '-L' + OMPI + '/lib -lmpi'

CUDA = '/usr/local/cuda'
CUDA_INCL = CUDA + '/include'
CUDA_LIB = '-L' + CUDA + '/lib64 -lm -lcuda -lcudart'

NCCL = '/home/gulhane.2/MCR_DL/nccl/build'
NCCL_INCL = NCCL + '/include'

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