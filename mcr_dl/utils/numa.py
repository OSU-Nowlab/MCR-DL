# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# return a list of list for cores to numa mapping
# [
#     [ cores for numa 0 ]
#     [ cores belong to numa 1 ]
#     ...
# ]

import distutils
import os
import psutil
import subprocess


# return a list of list for cores to numa mapping
# [
#     [ cores for numa 0 ]
#     [ cores belong to numa 1 ]
#     ...
# ]
def get_numa_cores():
    ret = []
    output = subprocess.check_output(['numactl', '--hardware']).decode("utf-8")
    lines = output.split('\n')
    for line in lines:
        if line.startswith('available:'):
            num_numas = int(line.split(' ')[1])
            break
    for numa in range(num_numas):
        for line in lines:
            if line.startswith(f'node {numa} cpus:'):
                cores = line.split(' ')[3:]
                ret.append([int(core) for core in cores])
    return ret
