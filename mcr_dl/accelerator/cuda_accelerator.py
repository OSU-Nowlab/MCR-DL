# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cuda
except ImportError:
    pass

# Delay import pynvml to avoid import error when CUDA is not available
pynvml = None


class CUDA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl'
        if pynvml is None:
            self._init_pynvml()

    def _init_pynvml(self):
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def communication_backend_name(self):
        return self._communication_backend_name

    def op_builder_dir(self):
        # try:
        #     # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
        #     # if successful this also means we're doing a local install and not JIT compile path
        #     from op_builder import __deepspeed__  # noqa: F401 # type: ignore
        #     return "op_builder"
        # except ImportError:
        #     return "mcr_dl.op_builder"
        return "mcr_dl.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict != None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None
