# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from mcr_dl.accelerator.abstract_accelerator import DeepSpeedAccelerator
# import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
import psutil
import os


# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'
        self._communication_backend_name = 'ccl'
        self.max_mem = psutil.Process().memory_info().rss

    def current_device_name(self):
        return 'cpu'

    def communication_backend_name(self):
        return self._communication_backend_name

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        # try:
        #     # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
        #     # if successful this also means we're doing a local install and not JIT compile path
        #     # from op_builder import __deepspeed__  # noqa: F401 # type: ignore
        #     from op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        # except ImportError:
        #     from deepspeed.ops.op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        from mcr_dl.op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder
