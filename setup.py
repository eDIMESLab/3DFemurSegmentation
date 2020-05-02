# -*- coding: utf-8 -*-
#!/usr/bin/env python

# python setup.py develop --user
# cython distance_matrix.pyx -a

try:
  from setuptools import setup
  from setuptools import Extension
  from setuptools import find_packages
except ImportError:
  from distutils.core import setup
  from distutils.core import Extension
  from distutils.core import find_packages

from Cython.Distutils import build_ext
from distutils.sysconfig import customize_compiler
import numpy
import os

class _build_ext (build_ext):
  '''
  Custom build type
  '''
  def build_extensions (self):
    customize_compiler(self.compiler)
    try:
      self.compiler.compiler_so.remove('-Wstrict-prototypes')
    except (AttributeError, ValueError):
      pass
    build_ext.build_extensions(self)

def read_description (readme_filename):
  '''
  Description package from filename
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

  except Exception:
    return ''


NAME = '3DFemurSegmentation'
DESCRIPTION = 'Python implementation of Graph-cut based 3D Segmentation of Femur'
URL = 'https://github.com/eDIMESLab/3DFemurSegmentation'
EMAIL = ['daniele.dallolio@unibo.it', 'nico.curti2@unibo.it']
AUTHOR = ["Daniele Dall'Olio", 'Nico Curti']
REQUIRES_PYTHON = '>=3.5'
VERSION = None
KEYWORDS = "graphcut maxflow femur segmentation"

# CPP_COMPILER = platform.python_compiler()
README_FILENAME = os.path.join(os.getcwd(), 'README.md')
# REQUIREMENTS_FILENAME =
# VERSION_FILENAME =

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except FileNotFoundError:
  LONG_DESCRIPTION = DESCRIPTION



ext_modules = [ Extension(name= '.'.join(['lib', 'fastDistMatrix']),
                          sources=[os.path.join(os.getcwd(), 'src', 'distance_matrix.pyx')],
                          libraries=[],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args = ['-g0',
                                                '-Ofast',
                                                '-Wno-unused-function',
                                                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
                          language='c++'
                          ),
                Extension(name= '.'.join(['lib', 'GraphCutSupport']),
                          sources=[os.path.join(os.getcwd(), 'maxflow-v3.01', 'graph.cpp'),
                                   os.path.join(os.getcwd(), 'maxflow-v3.01', 'maxflow.cpp'),
                                   os.path.join(os.getcwd(), 'src', 'graphcut.pyx')],
                          libraries=[],
                          library_dirs=[os.path.join('usr', 'lib'),
                                        os.path.join('usr', 'local', 'lib')],
                          include_dirs=[numpy.get_include(),
                                        os.path.join(os.getcwd(), 'include'),
                                        os.path.join(os.getcwd(), 'maxflow-v3.01')
                                        ],
                          extra_compile_args = ['-g0',
                                                '-Ofast',
                                                '-Wno-unused-function',
                                                '-Wno-write-strings',
                                                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
                          language='c++'
                          )
            ]

setup(
        name='3DSegmentationSupport',
        # version = ,
        description                   = DESCRIPTION,
        long_description              = LONG_DESCRIPTION,
        long_description_content_type = 'text/markdown',
        author                        = AUTHOR,
        author_email                  = EMAIL,
        maintainer                    = AUTHOR,
        maintainer_email              = EMAIL,
        # python_requires               = ,
        # install_requires              = get_requires(REQUIREMENTS_FILENAME),
        url                           = URL,
        download_url                  = URL,
        keywords                      = KEYWORDS,
        # packages                      = find_packages(),
        cmdclass = {'build_ext': _build_ext},
        license                       = 'MIT',
        ext_modules = ext_modules
      )
