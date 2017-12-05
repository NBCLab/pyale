# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
import os

from setuptools import setup

# fetch version from within neurosynth module
with open(os.path.join('pyale', 'version.py')) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(
    tests_require=['pytest']
    )

setup(
    name='pyale',
    version=__version__,
    description=('Python implementation of activation likelihood estimation. '
                 'Supports algorithm version {0}'.format(__algorithm__)),
    maintainer='Taylor Salo',
    maintainer_email='tsalo006@fiu.edu',
    install_requires=requirements,
    packages=['pyale'],
    license='',
    **extra_setuptools_args
)
