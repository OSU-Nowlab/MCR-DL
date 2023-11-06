# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .abstract_accelerator import DeepSpeedAccelerator

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.mps
except ImportError:
    pass


class MPS_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = "mps"
        self._communication_backend_name = None

    def current_device_name(self):
        return "mps:0"

    def communication_backend_name(self):
        return self._communication_backend_name

    def op_builder_dir(self):
        # try:
        #     # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
        #     # if successful this also means we're doing a local install and not JIT compile path
        #     from op_builder import __deepspeed__  # noqa: F401 # type: ignore

        #     return "op_builder"
        # except ImportError:
        #     return "deepspeed.ops.op_builder"
        return "mcr_dl.op_builder"

    # create an instance of op builder, specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None