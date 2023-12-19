import os
import yaml
import mcr_dl

class ConfigPath():
    def __init__(self, file_path = None):
        self.file_path = os.path.join(os.path.dirname(mcr_dl.__file__), "config.yml") if file_path is None else file_path
        print(self.file_path)
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