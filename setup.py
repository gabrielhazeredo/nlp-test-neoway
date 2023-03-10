from setuptools import setup, find_packages
import os
from os import path

# Model package name
NAME = 'nlp_test_neoway'
# Current Version
VERSION = os.environ.get('APP_VERSION', 'latest')

# Dependecies for the package
with open('requirements.txt') as r:
    DEPENDENCIES = [
        dep for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"),
                not dep.endswith("#dev"),
                len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    description='Sentiment Analysis based on B2W reviews',
    long_description=LONG_DESCRIPTION,
    author='Gabriel Hartmann de Azeredo',
    author_email='gabriel.hazeredo@gmail.com',
    license='MIT',
    packages=find_packages(exclude=("tests", "docs", "models", "analysis", "env")),
    entry_points={
        'console_scripts': [
            '{name}={name}.main:cli'.format(name=NAME)
        ],
    },
    # external packages as dependencies
    install_requires=DEPENDENCIES,
    python_requires='>=3',
)
