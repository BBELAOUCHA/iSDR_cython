from setuptools import find_packages, setup
from setuptools import Extension
from os import getenv
import numpy 
with open('README.md', mode='r') as file:
    long_description = file.read()

extensions = [Extension('iSDR_cython.cyISDR', ['src/iSDR_cython/cyISDR.pyx'],
                        define_macros=[('CYTHON_TRACE', 1)] if getenv('TESTING') == '1' else [],
                        include_dirs=[numpy.get_include()])]
setup(
    name='iSDR_cython',
    version='1.0.0',
    python_requires='~=3.5',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['numpy~=1.6', 'scipy~=1.4', 'scikit-learn~=0.22'],
    ext_modules=extensions,
    package_data={'iSDR_cython': ['*.pxd']},
    author='Brahim Belaoucha',
    author_email='brahim.belaoucha[at]gmail.com',
    description='A cython implemetation of iSDR source reconstruction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='Source reconstruction',
    url=' ',
)
