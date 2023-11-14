from setuptools import setup, find_packages

setup(
    name="MCR-DL",
    version="1",
    url="https://github.com/OSU-Nowlab/MCR-DL",
    packages=find_packages(where="."),
    package_dir={"mcr_dl": "mcr_dl"},
)