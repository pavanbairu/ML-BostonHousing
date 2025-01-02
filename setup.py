from setuptools import setup, find_packages

from typing import List

HYPHEN_E_DOT = "-e ."

def getrequirements(file_path: str) -> List[str]:
    """
        this function will return the list of packages to be installed
    """

    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        lines = [line.replace("\n", "") for line in lines]

        # if HYPHEN_E_DOT in lines:
        #     lines.remove(HYPHEN_E_DOT)
    
    return lines

setup(

    name = "ML-Project",
    version= "0.0.1",
    description= "Boston Housing",
    author= "Pavan",
    author_email= "pavan@gmail.com",
    packages= find_packages(),
    install_requires=getrequirements('requirements.txt')
)