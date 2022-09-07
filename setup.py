from setuptools import setup, find_packages
requirements = [
    "torch>=1.1.0",
    "scipy",
    "matplotlib",
    "numpy>=1.17",
]

setup(
    name='equivmed',
    version='0.11',
    author='Mauro Nascimben',
    author_email='msnascimben@gmail.com',
    description='Functions for clinical equivalence testing',
    url='https://github.com/m89p067/equivmed',
    packages=['equivmed'],
    install_requires=['requests'],
)
