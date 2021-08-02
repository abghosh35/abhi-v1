# Copyright 2020 Abhijit Ghosh. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# http://www.apache.org/licenses/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


"""Placeholder docstring"""
from __future__ import absolute_import

import os
from glob import glob
import sys

from setuptools import setup, find_packages


def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


# Declare minimal set for installation
required_packages = [
    'scikit-learn>=0.22.1',
    'joblib>=0.14.1',
    'torch',
    'tabulate',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'bs4'
]

setup(
    name="abhi",
    version=read_version(),
    description="Commonly used functions.",
    packages=find_packages(),
    # packages=['abhi'],
    # package_dir={"": "abhi"},
    # py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("abhi/*.py")],
    long_description=read("README.rst"),
    author="Abhijit Ghosh",
    author_email='abghosh35@gmail.com',
    url="https://github.com/abghosh35/PythonApp/",
    license="Apache License 2.0",
    keywords="Abhijit abhi ML Torch Model",
    install_requires=required_packages,
    include_package_data=True,
    # zip_safe=False
)