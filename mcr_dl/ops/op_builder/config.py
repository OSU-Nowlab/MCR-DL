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

import os
import yaml
import mcr_dl

class ConfigPath():
    def __init__(self, file_path = None):
        self.file_path = os.path.join(os.path.dirname(mcr_dl.__file__), "build_config.yml") if file_path is None else file_path
        self.config_data = self.load_config()
        self.mpi_path = self.config_data.get("mpi", {}).get("path")
        self.mpi_include = self.config_data.get("mpi", {}).get("include")
        self.cuda_path = self.config_data.get("cuda", {}).get("path")
        self.cuda_include = self.config_data.get("cuda", {}).get("include")
        self.nccl_path = self.config_data.get("nccl", {}).get("path")
        self.nccl_include = self.config_data.get("nccl", {}).get("include")

    def load_config(self):
        with open(self.file_path, "r") as file:
            config_data = yaml.safe_load(file)
            return config_data