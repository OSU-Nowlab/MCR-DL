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

from mcr_dl.cuda_accelerator import get_accelerator

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = get_accelerator().communication_backend_name()
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'mcr_dl'
DEFAULT_MAXSIZE = 24
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500