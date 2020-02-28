from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='abhi',
      version='0.1',
      description='Commonly used functions',
      url='https://github.com/abghosh35/',
      author='Abhijit Ghosh',
      author_email='abhijigh@amazon.com',
      license='GNU AGPL v3',
      packages=['abhi'],
      install_requires=[
          'joblib',
          'torch'
      ],
      include_package_data=True,
      zip_safe=False)