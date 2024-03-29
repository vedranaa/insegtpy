'''
Install with  pip install -e /path/to/insegtpy/folder/containing/this/file/
-e stands for editable,  so it will then create a link from the site-packages 
directory to the directory in which the code lives, meaning the latest version 
will be used without need to reinstall.

Following info from: https://stackoverflow.com/a/50468400 and
https://python-packaging.readthedocs.io/en/latest/index.html

'''
from setuptools import setup

setup(name='insegtpy',
    version='0.1',
    description='Interactive image segmentation.',
    url='https://github.com/vedranaa/insegtpy',
    author='Vedrana and Anders Dahl',
    author_email='{vand,abda}@dtu.dk',
    license='GNU GPSv3',
    packages=['insegtpy', 'insegtpy.annotators', 'insegtpy.models'],
    zip_safe=False,
    install_requires=['PyQt5', 'opencv-python', 'numpy', 'scikit-learn']
    )