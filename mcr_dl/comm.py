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

"""
    MCR-DL Communication Package: mcr_dl
    mcr_dl
        -- import and use mcr_dl
        -- use torch.distributed directly if both this package and torch.distributed use the same NCCL version
        -- use custom collectives
            -- can either use torch.dist or mcr_dl.ops.comm?
        Note: the old 1-bit compressed allreduce variants that resided in deepspeed.runtime.comm will be moved here as well.
    mcr_dl API
        -- must be kept fully compatible (same signatures) as torch.dist API to ensure backward/cross-framework compatibility.
        -- e.g. if a client code used
            import mcr_dl as dist
            instead of
            import torch.distributed as dist
            The code should work without breaking any of the public torch.distributed functionality
    Future:
        -- mcr_dl groups API should be brought into ds.comm
"""

import torch
import os
from datetime import timedelta

from mcr_dl import utils
from mcr_dl.utils.comms_logging import CommsLogger
from mcr_dl.utils import timer, get_caller_func

from .constants import TORCH_DISTRIBUTED_DEFAULT_PORT, default_pg_timeout
from .constants import *
from .reduce_op import *
from .cuda_accelerator import get_accelerator
from .nccl import NCCLBackend
from .mpi import MPIBackend
from .torch import TorchBackend

# Current mcr-dl backend (cdb) global object for simple access by client code
cdb = None

# Create global timer for ops
timers = timer.SynchronizedWallClockTimer()
timer_summary = {}

comms_logger = CommsLogger()

# Maintain objects of all initialized ds backends and assign them using the API functions in this file
nccl_backend = None
mpi_backend = None
ccl_backend = None
hccl_backend = None

# This should be set here so all rank/size information from the launcher can be propagated
from mcr_dl.utils import *


class ProcessGroup():

    def __init__(self, comm_id, ranks=[]):
        self.ranks = ranks
        self.comm_id = comm_id
        self.size = len(ranks)

def _configure_using_config_file(config):
    if config.comms_logger_enabled:
        comms_logger.configure(config)


def configure(
    mcr_dl_config=None,
    enabled=None,
    prof_all=None,
    prof_ops=None,
    verbose=None,
    debug=None,
):

    if mcr_dl_config is not None:
        _configure_using_config_file(mcr_dl_config.comms_config)

    if enabled is not None:
        comms_logger.enabled = enabled

    if prof_all is not None:
        comms_logger.prof_all = prof_all

    if prof_ops is not None:
        comms_logger.prof_ops = prof_ops

    if verbose is not None:
        comms_logger.verbose = verbose

    if debug is not None:
        comms_logger.debug = debug

# Logging wrapper for timing ops
def timed_op(func):

    def log_wrapper(*args, **kwargs):
        # Add enabled flag so that overhead to each comm op is two if conditions at most
        if comms_logger.enabled:
            if ('prof' in kwargs
                    and kwargs['prof']) or comms_logger.prof_all or ('log_name' in kwargs
                                                                     and kwargs['log_name'] in comms_logger.prof_ops):
                # Need func args for their defaults
                func_args = get_default_args(func)
                func_args.update(kwargs)
                msg_size = get_msg_size_from_args(func, *args, **kwargs)
                log_name = get_debug_log_name(func_args, comms_logger.debug)
                timers(log_name).start()
        # Return the op, then stop the op's timer
        try:
            return func(*args, **kwargs)
        finally:
            if comms_logger.enabled:
                # Need to make op blocking for accurate logging
                get_accelerator().synchronize()
                # If we're using MPI, we can't simply sync the stream
                if cdb.using_mpi:
                    cdb.barrier()
                if ('prof' in kwargs and kwargs['prof']) or comms_logger.prof_all or (
                        'log_name' in kwargs and kwargs['log_name'] in comms_logger.prof_ops):
                    log_name = get_debug_log_name(func_args, comms_logger.debug)
                    raw_name = func.__name__
                    timers(log_name).stop()
                    # need temp var since 'elapsed' resets events
                    time_elapsed = timers(log_name).elapsed(reset=False)
                    comms_logger.append(raw_name, log_name, time_elapsed, msg_size)

    return log_wrapper


# For compatibility with torch distributed's init_process_group, we shall retain the signature from PyTorch code.
# MCR-DL NCCL/MPI backend may not need all these params as we will have our own implementation.
# Please read full torch.distributed API docs from https://pytorch.org/docs/stable/distributed.html


# UNUSED: Future helper function to initialize DS backends
def init_mcr_dl_backend(ds_backend, timeout, init_method):
    global cdb
    global nccl_backend
    global mpi_backend
    global ccl_backend
    global hccl_backend

    rank = int(os.getenv('RANK', '-1'))
    size = int(os.getenv('WORLD_SIZE', '-1'))

    if ds_backend == NCCL_BACKEND:
        utils.logger.debug("NCCL backend in MCR-DL not yet implemented")
    elif ds_backend == MPI_BACKEND:
        utils.logger.debug("MPI backend in MCR-DL not yet implemented")
    else:
        utils.logger.debug(f"MCR-DL does not support {ds_backend} backend")


def is_initialized():
    #assert cdb is not None, 'MCR-DL backend not set, please initialize it using init_process_group()'
    if cdb is None:
        return False
    else:
        return cdb.is_initialized()


def destroy_process_group(group=None):
    global cdb
    return cdb.destroy_process_group(group=group)


def new_group(ranks):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.new_group(ranks)


def is_available() -> bool:

    # Returns ``True`` if the mcr-dl comm package is available.

    # TODO: load other ops. Clients including mcr-dl itself should use mcr-dl to import
    # any communication related primitives from this package.
    # use hasattr(mcr-dl.csrc.ops, "_comm") or something
    return True


def set_backend():
    global cdb
    global nccl_backend
    global mpi_backend
    global ccl_backend
    global hccl_backend

    backend_name = get_accelerator().communication_backend_name()

    if backend_name == NCCL_BACKEND:
        if nccl_backend is not None and nccl_backend.is_initialized():
            cdb = nccl_backend
    elif backend_name == MPI_BACKEND:
        if mpi_backend is not None and mpi_backend.is_initialized():
            cdb = mpi_backend
    elif backend_name == CCL_BACKEND:
        if ccl_backend is not None and ccl_backend.is_initialized():
            cdb = ccl_backend
    elif backend_name == HCCL_BACKEND:
        if hccl_backend is not None and hccl_backend.is_initialized():
            cdb = hccl_backend


@timed_op
def broadcast(tensor, src, op=ReduceOp.SUM, group=None, async_op=False, prof=False, log_name='broadcast', debug=get_caller_func()):
    global cdb
    return cdb.broadcast(tensor=tensor, src=src, op=op, group=group, async_op=async_op)


@timed_op
def all_gather(tensor_list,
               tensor,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_gather',
               debug=get_caller_func()):
    global cdb
    return cdb.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)


def has_reduce_scatter_tensor():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.has_reduce_scatter_tensor()


def reduce_scatter_fn(output_tensor,
                      tensor,
                      op=ReduceOp.SUM,
                      group=None,
                      async_op=False,
                      prof=False,
                      debug=get_caller_func()):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    if cdb.has_reduce_scatter_tensor():
        return reduce_scatter_tensor(output_tensor,
                                     tensor,
                                     op=op,
                                     group=group,
                                     async_op=async_op,
                                     prof=prof,
                                     debug=debug)
    else:
        if get_rank() == 0:
            utils.logger.warning_once("unable to find torch.distributed.reduce_scatter_tensor. will fall back to "
                                      "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                      "please consider upgrading your pytorch installation.")
        input_tensor_lst = list(torch.chunk(tensor, cdb.get_world_size(group)))
        return reduce_scatter(output_tensor,
                              input_tensor_lst,
                              op=op,
                              group=group,
                              async_op=async_op,
                              prof=prof,
                              debug=debug)


@timed_op
def reduce_scatter_tensor(output_tensor,
                          tensor,
                          op=ReduceOp.SUM,
                          group=None,
                          async_op=False,
                          prof=False,
                          log_name='reduce_scatter_tensor',
                          debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter_tensor(output_tensor=output_tensor,
                                     input_tensor=tensor,
                                     op=op,
                                     group=group,
                                     async_op=async_op)


@timed_op
def all_gather_into_tensor(output_tensor,
                           tensor,
                           group=None,
                           async_op=False,
                           prof=False,
                           log_name='all_gather_into_tensor',
                           debug=get_caller_func()):
    global cdb
    return cdb.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=tensor, group=group, async_op=async_op)


def has_all_gather_into_tensor():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.has_all_gather_into_tensor()


def allgather_fn(output_tensor, input_tensor, group=None, async_op=False, debug=get_caller_func()):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    if cdb.has_all_gather_into_tensor():
        return all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op, debug=debug)
    else:
        if get_rank() == 0:
            utils.logger.warning_once("unable to find torch.distributed.all_gather_into_tensor. will fall back to "
                                      "torch.distributed.all_gather which will result in suboptimal performance. "
                                      "please consider upgrading your pytorch installation.")
        output_tensors = list(torch.chunk(output_tensor, cdb.get_world_size(group)))
        return all_gather(output_tensors, input_tensor, group=group, async_op=async_op, debug=debug)


@timed_op
def all_to_all_single(output,
                      tensor,
                      output_split_sizes=None,
                      input_split_sizes=None,
                      op=ReduceOp.SUM,
                      group=None,
                      async_op=False,
                      prof=False,
                      log_name='all_to_all_single',
                      debug=get_caller_func()):
    global cdb
    return cdb.all_to_all_single(output=output,
                                 input=tensor,
                                 output_split_sizes=output_split_sizes,
                                 input_split_sizes=input_split_sizes,
                                 op=op,
                                 group=group,
                                 async_op=async_op)


@timed_op
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    global cdb
    return cdb.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)


@timed_op
def send(tensor, dst, group=None, tag=0, prof=False, log_name='send', debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def recv(tensor, src=None, group=None, tag=0, prof=False, log_name='recv', debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def isend(tensor, dst, group=None, tag=0, prof=False, log_name='isend', debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def irecv(tensor, src=None, group=None, tag=0, prof=False, log_name='irecv', debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def gather(tensor,
           gather_list=None,
           dst=0,
           group=None,
           async_op=False,
           prof=False,
           log_name='gather',
           debug=get_caller_func()):
    global cdb
    return cdb.gather(tensor=tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)


@timed_op
def scatter(tensor,
            scatter_list=None,
            src=0,
            group=None,
            async_op=False,
            prof=False,
            log_name='scatter',
            debug=get_caller_func()):
    global cdb
    return cdb.scatter(tensor=tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)


@timed_op
def barrier(group=None, async_op=False, device_ids=None, prof=False, log_name='barrier', debug=get_caller_func()):
    global cdb
    return cdb.barrier(group=group, async_op=async_op)


@timed_op
def monitored_barrier(group=None,
                      timeout=None,
                      wait_all_ranks=False,
                      prof=False,
                      log_name='monitored_barrier',
                      debug=get_caller_func()):
    global cdb
    return cdb.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)


def log_summary(show_straggler=False):
    global cdb
    barrier(log_name='log_summary_barrier')
    if cdb.get_rank() == 0:
        comms_logger.log_all(print_log=True, show_straggler=show_straggler)
    else:
        comms_logger.log_all(print_log=False, show_straggler=show_straggler)
    barrier(log_name='log_summary_barrier')


@timed_op
def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           group=None,
           async_op=False,
           prof=False,
           log_name='reduce',
           debug=get_caller_func()):
    global cdb
    return cdb.reduce(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)


@timed_op
def reduce_scatter(output,
                   input_list,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   prof=False,
                   log_name='reduce_scatter',
                   debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter(output=output, input_list=input_list, op=op, group=group, async_op=async_op)


def has_all_reduce_coalesced():
    """"""
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    assert cdb.has_all_reduce_coalesced is not None, 'has_all_reduce_coalesced is not yet defined'
    return cdb.has_all_reduce_coalesced


def has_coalescing_manager():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    assert cdb.has_coalescing_manager is not None, 'has_coalescing_manager is not yet defined'
    return cdb.has_coalescing_manager


def all_gather_coalesced(output_tensors, input_tensors, group=None, async_op=False):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.all_gather_coalesced(output_tensors, input_tensors, group=group, async_op=async_op)


@timed_op
def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_reduce',
               debug=get_caller_func()):
    #if profile_comm:
    # context of the timers?
    # timers.start()
    # TensorBoard logging for comm calls.?
    global cdb
    #print(f'op = {op}, cdb= {cdb.name}')
    return cdb.all_reduce(tensor, op, group, async_op)


@timed_op
def inference_all_reduce(tensor,
                         op=ReduceOp.SUM,
                         group=None,
                         async_op=False,
                         prof=False,
                         log_name='all_reduce',
                         debug=get_caller_func()):
    global cdb
    return cdb.inference_all_reduce(tensor, op, group, async_op)


@timed_op
def all_reduce_coalesced(tensors,
                         op=ReduceOp.SUM,
                         group=None,
                         async_op=False,
                         prof=False,
                         log_name='all_reduce',
                         debug=get_caller_func()):
    global cdb
    return cdb.all_reduce_coalesced(tensors, op, group, async_op)


def get_world_group():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.get_world_group()


def get_world_size(group=None) -> int:
    """
    Returns the number of processes in the current process group
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The world size of the process group
        -1, if not part of the group
    """
    global cdb

    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.get_world_size(group)


def get_rank(group=None):
    """
    Returns the rank of the current process in the provided ``group`` or the
    default group if none was provided.
    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The rank of the process group
        -1, if not part of the group
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.get_rank(group)


def get_local_rank():
    """
        Helper function to get local rank after a backend has been set and initialized
        Args:
            None
        Returns:
            local rank (= GPU device ID)
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return get_local_rank_from_launcher()


def get_global_rank(group=None, group_rank=0):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    return cdb.get_global_rank(group, group_rank)


def get_all_ranks_from_group(group=None):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'MCR-DL backend not set, please initialize it using init_process_group()'
    rank = 0
    group_ranks = []
    try:
        while True:
            group_ranks.append(cdb.get_global_rank(group, rank))
            rank += 1
    except RuntimeError:
        pass
    return group_ranks


# Main MCR-DL Comms. public API.
def init_distributed(dist_backend=None,
                     auto_mpi_discovery=True,
                     distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT,
                     verbose=True,
                     timeout=default_pg_timeout,
                     init_method=None,
                     dist_init_required=None,
                     config=None,
                     rank=-1,
                     world_size=-1,
                     use_mcr_dl=False):
    ''' Initialize dist backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). torch distributed backend, e.g., nccl, mpi, gloo
        auto_mpi_discovery Optional (bool). if distributed environment variables are not set, attempt to discover them from MPI
        distributed_port: Optional (int). torch distributed backend port
        verbose: Optional (bool). verbose logging
        timeout: Optional (timedelta). Timeout for operations executed against the process group. Default value equals 30 minutes.
        init_method: Optional (string). Torch distributed, URL specifying how to initialize the process group. Default is “env://” if no init_method or store is specified.
        config: Optional (dict). MCR-DL configuration for setting up comms options (e.g. Comms profiling)
        rank: Optional (int). The current manually specified rank. Some init_method like “tcp://” need the rank and world_size as well (see: https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
        world_size: Optional (int). Desired world_size for the TCP or Shared file-system initialization.
    '''
    global cdb

    configure(mcr_dl_config=config)

    if dist_init_required is None:
        dist_init_required = cdb is None or not cdb.is_initialized()

    if cdb is None:
        init_mcr_dl_backend(get_accelerator().communication_backend_name(), timeout, init_method)
        set_backend()
        utils.logger.info(f'cdb={cdb}')

    if cdb is None and torch.distributed.is_initialized():
        # The user initialized torch.dist themselves, create cdb and short-circuit
        cdb = TorchBackend(dist_backend, timeout, init_method)
        return
    if dist_init_required is False:
        assert (
            cdb is not None and cdb.is_initialized() is True
        ), "Distributed backend is not initialized. Please set dist_init_required to True or initialize before calling mcr_dl.initialize()"
    else:
        # Initialize torch distributed if needed
        required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
        if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
            if verbose:
                utils.logger.info("Not using the MCR-DL or dist launchers, attempting to detect MPI environment...")
            if in_aml() and not in_dlts():
                patch_aml_env_for_torch_nccl_backend(verbose=verbose)
            elif in_aws_sm():
                patch_aws_sm_env_for_torch_nccl_backend(verbose=verbose)
            else:
                mpi_discovery(distributed_port=distributed_port, verbose=verbose)
        if cdb is not None and cdb.is_initialized():
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.info('Distributed backend already initialized')
        else:
            assert isinstance(timeout, timedelta)
            if use_mcr_dl:
                if dist_backend == 'nccl':
                    if int(os.getenv('RANK', '0')) == 0:
                        utils.logger.info('Initializing NCCLBackend in MCR-DL')
                    cdb = NCCLBackend()
                elif dist_backend == 'mpi' :
                    assert True, "Pure MPIBackend is not supported."
                    if int(os.getenv('RANK', '0')) == 0:
                        utils.logger.info('Initializing MPIBackend in MCR-DL')
                    cdb = MPIBackend()
            else:
                # Create a torch backend object, initialize torch distributed, and assign to cdb
                if int(os.getenv('RANK', '0')) == 0:
                    utils.logger.info(
                        'Initializing TorchBackend in MCR-DL with backend {}'.format(
                            dist_backend))
                cdb = TorchBackend(dist_backend, timeout, init_method)


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    '''
    Discovery MPI environment via mpi4py and map to relevant dist state
    '''
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        utils.logger.info(
            "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}".
            format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                   os.environ['MASTER_PORT']))

    if cdb is not None and cdb.is_initialized():
        assert cdb.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(rank, cdb.get_rank())
        assert cdb.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, cdb.get_world_size())


def in_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_aws_sm():
    # Are we running inside an AWS SageMaker environment?
    return 'SM_TRAINING_ENV' in os.environ


def in_dlts():
    # Are we running on a DLTS cluster?
    return 'DLTS_JOB_ID' in os.environ


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(os.environ["WORLD_SIZE"])

    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = DEFAULT_AML_MASTER_PORT

    if verbose:
        utils.logger.info("NCCL_SOCKET_IFNAME original value = {}".format(os.environ["NCCL_SOCKET_IFNAME"]))

    os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_AML_NCCL_SOCKET_IFNAME
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    if verbose:
        utils.logger.info(
            "Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))


def patch_aws_sm_env_for_torch_nccl_backend(verbose=True):
    """Helper routine to get and set environment variables when running inside an AWS SageMaker environment.
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

    if verbose:
        utils.logger.info(
            "Discovered AWS SageMaker settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))
