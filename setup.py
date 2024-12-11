from setuptools import setup, find_packages

setup(
    name='lora_ga_init',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)