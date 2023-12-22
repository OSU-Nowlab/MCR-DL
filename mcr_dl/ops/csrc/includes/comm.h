/*
* The MVAPICH software package is developed by the team members of
* The Ohio State University's Network-Based Computing Laboratory (NBCL),
* headed by Professor Dhabaleswar K. (DK) Panda.
*
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

*     http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// enum class ReduceOp {
//     SUM = 0,
//     AVG,
//     PRODUCT,
//     MIN,
//     MAX,
//     BAND,  // Bitwise AND
//     BOR,   // Bitwise OR
//     BXOR,  // Bitwise XOR
//     UNUSED,
// };

// void create_comm_group(std::vector<int> comm_ranks, int rank, int comm_id, int color);

#define MPICHECK(cmd)                                                        \
    do {                                                                     \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS) {                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                                            \
    do {                                                                                          \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                           \
    do {                                                                                         \
        ncclResult_t ret = cmd;                                                                  \
        if (ret != ncclSuccess) {                                                                \
            printf(                                                                              \
                "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(ret)); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CUDA_STREAM_SYNCHRONIZE(_nccl_stream)                                            \
    do {                                                                                 \
        cudaError_t err = cudaErrorNotReady;                                             \
        int flag;                                                                        \
        while (err == cudaErrorNotReady) {                                               \
            err = cudaStreamQuery(_nccl_stream);                                         \
            MPICHECK(MPI_Iprobe(                                                         \
                MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE)); \
        }                                                                                \
        CUDACHECK(err);                                                                  \
    } while (0)