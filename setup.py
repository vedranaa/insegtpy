'''
Install with  pip install -e /path/to/insegtpy/folder/containing/this/file/
-e stands for editable,  so it will then create a link from the site-packages
directory to the directory in which the code lives, meaning the latest version
will be used without need to reinstall.

Following info from: https://stackoverflow.com/a/50468400 and
https://python-packaging.readthedocs.io/en/latest/index.html

'''
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext


km_dict_module = Extension(
    'insegtpy.models.km_dict',
    sources=['insegtpy/models/km_dict.cpp'],
)


class my_build_ext(build_ext):
    # Hack to remove PyInit_km_dict from the exported symbols.
    # Needed to avoid link errors on Windows where setuptools tries to manually
    # export this symbol.
    # Normally, this function is needed to build a Python extension module, but
    # since we are interfacing the the code via ctypes, we don't need it.
    def get_export_symbols(self, ext):
        export_symbols = super().get_export_symbols(ext)
        new_export_symbols = []
        for symbol in export_symbols:
            if symbol.startswith('PyInit_'):
                continue
            new_export_symbols.append(symbol)
        ext.export_symbols = new_export_symbols
        return new_export_symbols


setup(name='insegtpy',
    version='0.1',
    description='Interactive image segmentation.',
    url='https://github.com/vedranaa/insegtpy',
    author='Vedrana and Anders Dahl',
    author_email='{vand,abda}@dtu.dk',
    license='GNU GPSv3',
    packages=['insegtpy', 'insegtpy.annotators', 'insegtpy.models'],
    zip_safe=False,
    install_requires=['PyQt5', 'opencv-python-headless', 'numpy', 'scikit-learn'],
    ext_modules=[km_dict_module],
    cmdclass={'build_ext': my_build_ext},
)