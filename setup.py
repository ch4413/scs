from distutils.core import setup
from setuptools import setup, find_packages

setup(name='scsavailability',
      version='0.1',
      description='Python SCS Availability',
      author='Chris Hughes',
      author_email='christopher.hughes@bath.edu',
      include_package_data=True,
      package_data={'': ['data/*.csv','data/sql/*']},
      url='https://www.python.org',
      packages=find_packages()
     )
     