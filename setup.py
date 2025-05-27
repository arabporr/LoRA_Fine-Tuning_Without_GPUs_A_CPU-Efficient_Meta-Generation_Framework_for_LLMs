from setuptools import setup, find_packages

from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any '-e .' entries and comments.
    """
    with open(file_path) as file:
        lines = file.readlines()

    # Remove any '-e .' entries and comments
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and line != "-e .":
            requirements.append(line)

    return requirements


setup(
    name="LoRA",
    version="1.0.0",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="Low-Rank Adaptation (LoRA) of Large Language Models on CPU",
)
