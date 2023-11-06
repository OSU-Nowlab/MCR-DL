# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import abc
from abc import ABC


class DeepSpeedAccelerator(ABC):

    def __init__(self):
        self._name = None
        self._communication_backend_name = None

    @abc.abstractmethod
    def current_device_name(self):
        ...

    @abc.abstractmethod
    def communication_backend_name(self):
        ...

    # create an instance of op builder, specified by class_name
    @abc.abstractmethod
    def create_op_builder(self, class_name):
        ...
