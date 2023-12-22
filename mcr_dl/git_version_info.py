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

try:
    #  This is populated by setup.py
    from .git_version_info_installed import *  # noqa: F401 # type: ignore
except ModuleNotFoundError:
    import os
    if os.path.isfile('version.txt'):
        # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
        version = open('version.txt', 'r').read().strip()
    else:
        version = "0.0.0"
    git_hash = '[none]'
    git_branch = '[none]'

    from .ops.op_builder.all_ops import ALL_OPS
    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    torch_info = {'version': "0.0", "cuda_version": "0.0", "hip_version": "0.0"}