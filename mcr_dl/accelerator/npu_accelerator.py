# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import importlib
import inspect

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.npu
except ImportError:
    pass


class NPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        super().__init__()
        self._name = 'npu'
        self._communication_backend_name = 'hccl'
        # dict that holds class name <--> class type mapping i.e.
        # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
        # this dict will be filled at init stage
        self.class_dict = None

    def current_device_name(self):
        return 'npu:{}'.format(torch.npu.current_device())

    def communication_backend_name(self):
        return self._communication_backend_name

    def op_builder_dir(self):
        # try:
        #     # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
        #     # if successful this also means we're doing a local install and not JIT compile path
        #     from op_builder import __deepspeed__  # noqa: F401 # type: ignore
        #     return "op_builder.npu"
        # except ImportError:
        #     return "mcr_dl.op_builder.npu"
        return "mcr_dl.op_builder.npu"

    def _lazy_init_class_dict(self):
        if self.class_dict:
            return

        op_builder_module = importlib.import_module(self.op_builder_dir())

        # get op builder class from op_builder/npu/__init__.py
        self.class_dict = {}
        for class_name, class_obj in inspect.getmembers(op_builder_module, inspect.isclass):
            self.class_dict[class_name] = class_obj

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        builder_class = self.get_op_builder(class_name)
        return None if builder_class is None else builder_class()

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return self.class_dict['NotImplementedBuilder'] if 'NotImplementedBuilder' in self.class_dict else None
