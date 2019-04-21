# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


long_description = open("README.md").read()

install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'elephant',
    'quantities',
    'neo'
]

extras_require = {
    'testing': ['pytest']
}

setup(
    name="visualstimulation",
    install_requires=install_requires,
    tests_require=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    version='0.1',
)
