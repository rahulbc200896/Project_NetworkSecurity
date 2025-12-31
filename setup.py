from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_name:str)->List[str]:
    requirements = []
    with open(file_name,"r") as file:
        requirements = file.readlines()

        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="Network Security",
    version="0.0.1",
    author="Rahul B C",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)