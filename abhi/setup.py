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


from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='abhi',
      version='0.1 (beta)',
      description='Commonly used functions',
      url='https://github.com/abghosh35/',
      author='Abhijit Ghosh',
      author_email='abhijigh@amazon.com',
      license='GNU AGPL v3',
      packages=['abhi'],
      install_requires=[
          'scikit-learn>=0.22.1',
          'joblib',
          'torch',
          'tabulate'
      ],
      include_package_data=True,
      zip_safe=False)