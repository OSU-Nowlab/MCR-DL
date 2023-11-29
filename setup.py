
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
MCR-DL Setup
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
import time
import typing

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

from mcr_dl.ops.op_builder import get_default_compute_capabilities
from mcr_dl.ops.op_builder.all_ops import ALL_OPS


RED_START = '\033[31m'
RED_END = '\033[0m'
ERROR = f"{RED_START} [ERROR] {RED_END}"

def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]

install_requires = fetch_requirements('requirements/requirements.txt')
extras_require = {
    '1bit_mpi': fetch_requirements('requirements/requirements-1bit-mpi.txt'),
}

# Make an [all] extra that installs all needed dependencies.
all_extras = set()
for extra in extras_require.items():
    for req in extra[1]:
        all_extras.add(req)
extras_require['all'] = list(all_extras)

if torch_available:
    TORCH_MAJOR = torch.__version__.split('.')[0]
    TORCH_MINOR = torch.__version__.split('.')[1]
else:
    TORCH_MAJOR = "0"
    TORCH_MINOR = "0"

if torch_available and not torch.cuda.is_available():
    # Fix to allow docker builds, similar to https://github.com/NVIDIA/apex/issues/486.
    print("[WARNING] Torch did not find cuda available, if cross-compiling or running with cpu only "
          "you can ignore this message. Adding compute capability for Pascal, Volta, and Turing "
          "(compute capabilities 6.0, 6.1, 6.2)")
    if not bool(os.environ.get("TORCH_CUDA_ARCH_LIST", None)):
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_default_compute_capabilities()

# Default to pre-install kernels to false so we rely on JIT on Linux, opposite on Windows.
if sys.platform == "win32":
    assert torch_available, "Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops."


def command_exists(cmd):
    if sys.platform == "win32":
        result = subprocess.Popen(f'{cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 1
    else:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0


compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
install_ops = dict.fromkeys(ALL_OPS.keys(), False)

# Write out version/git info.
git_hash_cmd = "git rev-parse --short HEAD"
git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
if command_exists('git'):
    try:
        result = subprocess.check_output(git_hash_cmd, shell=True)
        git_hash = result.decode('utf-8').strip()
        result = subprocess.check_output(git_branch_cmd, shell=True)
        git_branch = result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
        git_branch = "unknown"
else:
    git_hash = "unknown"
    git_branch = "unknown"

# Parse the MCR-DL version string from version.txt.
version_str = open('version.txt', 'r').read().strip()

# add git hash to version.
version_str += f'+{git_hash}'

torch_version = ".".join([TORCH_MAJOR, TORCH_MINOR])
bf16_support = False
# Set cuda_version to 0.0 if cpu-only.
cuda_version = "0.0"
nccl_version = "0.0"
# Set hip_version to 0.0 if cpu-only.
hip_version = "0.0"
if torch_available and torch.version.cuda is not None:
    cuda_version = ".".join(torch.version.cuda.split('.')[:2])
    if sys.platform != "win32":
        if isinstance(torch.cuda.nccl.version(), int):
            # This will break if minor version > 9.
            nccl_version = ".".join(str(torch.cuda.nccl.version())[:2])
        else:
            nccl_version = ".".join(map(str, torch.cuda.nccl.version()[:2]))
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_available():
        bf16_support = torch.cuda.is_bf16_supported()
if torch_available and hasattr(torch.version, 'hip') and torch.version.hip is not None:
    hip_version = ".".join(torch.version.hip.split('.')[:2])
torch_info = {
    "version": torch_version,
    "bf16_support": bf16_support,
    "cuda_version": cuda_version,
    "nccl_version": nccl_version,
    "hip_version": hip_version
}

print(f"version={version_str}, git_hash={git_hash}, git_branch={git_branch}")
with open('mcr_dl/git_version_info_installed.py', 'w') as fd:
    fd.write(f"version='{version_str}'\n")
    fd.write(f"git_hash='{git_hash}'\n")
    fd.write(f"git_branch='{git_branch}'\n")
    fd.write(f"installed_ops={install_ops}\n")
    fd.write(f"compatible_ops={compatible_ops}\n")
    fd.write(f"torch_info={torch_info}\n")

print(f'install_requires={install_requires}')
print(f'compatible_ops={compatible_ops}')

# Parse README.md to make long_description for PyPI page.
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
    readme_text = fin.read()

start_time = time.time()

setup(name='mcr_dl',
      version=version_str,
      description='MCR-DL library',
      long_description=readme_text,
      long_description_content_type='text/markdown',
      author='NOWLAB Team',
      author_email='https://hidl.cse.ohio-state.edu',
      project_urls={
        #   'Documentation': 'https://MCR_DL.readthedocs.io',
          'Source': 'https://github.com/OSU-Nowlab/MCR-DL',
      },
      install_requires=install_requires,
      extras_require=extras_require,
      packages=find_packages(include=['mcr_dl', 'mcr_dl.*']),
      include_package_data=True,
      license='Apache Software License 2.0')

end_time = time.time()
print(f'MCR-DL build time = {end_time - start_time} secs')
