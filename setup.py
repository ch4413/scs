from distutils.core import setup

setup(name='scsavailability',
      version='0.1',
      description='Python SCS Availability',
      author='Chris Hughes',
      author_email='christopher.hughes@bath.edu',
      url='https://www.python.org',
      packages=['scsavailability',
               'scsavailability.data',
               'scsavailability.test']
     )