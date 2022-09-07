from setuptools import setup, find_packages
#__version__ = '0.11'
requirements = [
    "scipy>=1.8.0",
    "numpy>=1.22.1",
    "matplotlib>=3.4.2",
    "seaborn>=0.11.1",
    "statsmodels>=0.13.2",
]
test_requirements = ["pytest>=3"]
#version = __version__
setup(
    name='equivmed',
    version='0.11',
    author='Mauro Nascimben',
    author_email='msnascimben@gmail.com',
    python_requires=">=3.6",
    description='Functions for clinical equivalence testing',
    url='https://github.com/m89p067/equivmed',
    packages=find_packages(include=["equivmed", "equivmed.*"]),
    install_requires=requirements,
    #version=__version__,
)
