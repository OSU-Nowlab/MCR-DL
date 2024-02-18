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
from datetime import timedelta

NCCL_BACKEND = 'nccl'
CCL_BACKEND = 'ccl'
MPI_BACKEND = 'mpi'
GLOO_BACKEND = 'gloo'
SCCL_BACKEND = 'sccl'
HCCL_BACKEND = 'hccl'

DEFAULT_AML_MASTER_PORT = "54965"
DEFAULT_AML_NCCL_SOCKET_IFNAME = "^docker0,lo"

#########################################
# Comms Logger
#########################################
# Comms Logger. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
COMMS_LOGGER_FORMAT = '''
The Comms Logger can be specified as:
"comms_logger": {
  "enabled": true,
  "verbose": false,
  "prof_all": true,
  "debug": false,
  "prof_ops": ["all_reduce", "custom_all_reduce_name"]
}
'''
COMMS_LOGGER = "comms_logger"

# Comms logger enable signal
COMMS_LOGGER_ENABLED = "enabled"
COMMS_LOGGER_ENABLED_DEFAULT = False

# Comms logger verbose signal
COMMS_LOGGER_VERBOSE = "verbose"
COMMS_LOGGER_VERBOSE_DEFAULT = False

# comms logger profile all ops signal
COMMS_LOGGER_PROF_ALL = "prof_all"
COMMS_LOGGER_PROF_ALL_DEFAULT = True

# comms logger show all ops signal
COMMS_LOGGER_DEBUG = "debug"
COMMS_LOGGER_DEBUG_DEFAULT = False

# comms logger profile specific ops in list
COMMS_LOGGER_PROF_OPS = "prof_ops"
COMMS_LOGGER_PROF_OPS_DEFAULT = []


#############################################
# Torch distributed constants
#############################################
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
# (only if NCCL_BLOCKING_WAIT or NCCL_ASYNC_ERROR_HANDLING is set to 1).
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout = timedelta(minutes=int(os.getenv("MCR-DL_TIMEOUT", default=30)))
INFERENCE_GENERIC_MODE = 'generic'
INFERENCE_SPECIALIZED_MODE = 'specialized'

