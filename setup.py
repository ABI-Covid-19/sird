from setuptools import setup, find_packages

# Deploy our package.

setup(
    author='Alan Garny',
    author_email='a.garny@auckland.ac.nz',
    description='A Python package to model Covid-19 using the SIRD model',
    install_requires=[
        'bs4',
        'filterpy',
        'matplotlib',
        'numpy',
        'pandas',
        'requests',
    ],
    license='Apache 2.0',
    name='sird',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/ABI-Covid-19/sird',
    version='0.1.0',
)
