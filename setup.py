from setuptools import setup, find_packages
from typing import List

hyp_e_dot='-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    it will return the list of requirements
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements = [req.replace('\n', '') for req in requirements ]
    if hyp_e_dot in requirements:
        requirements.remove(hyp_e_dot)
    return requirements


setup(
name='customer_churn',
version='0.1.0',
author='JEEL',
author_email='jeel15thummar@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
  
)