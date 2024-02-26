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

import sys, os

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from utils import *
from all_reduce import run_all_reduce
from all_gather import run_all_gather
from all_to_all import run_all_to_all
from pt2pt import run_pt2pt
from broadcast import run_broadcast
from constants import *


# For importing
def main(args, rank):

    mcr_dl.init_processes(args.dist, args.backend)

    ops_to_run = []
    if args.all_reduce:
        ops_to_run.append('all_reduce')
    if args.all_gather:
        ops_to_run.append('all_gather')
    if args.broadcast:
        ops_to_run.append('broadcast')
    if args.pt2pt:
        ops_to_run.append('pt2pt')
    if args.all_to_all:
        ops_to_run.append('all_to_all')

    if len(ops_to_run) == 0:
        ops_to_run = ['all_reduce', 'all_gather', 'all_to_all', 'broadcast', 'pt2pt']

    for comm_op in ops_to_run:
        if comm_op == 'all_reduce':
            run_all_reduce(local_rank=rank, args=args)
        if comm_op == 'all_gather':
            run_all_gather(local_rank=rank, args=args)
        if comm_op == 'all_to_all':
            run_all_to_all(local_rank=rank, args=args)
        if comm_op == 'pt2pt':
            run_pt2pt(local_rank=rank, args=args)
        if comm_op == 'broadcast':
            run_broadcast(local_rank=rank, args=args)


# For directly calling benchmark
if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    main(args, rank)