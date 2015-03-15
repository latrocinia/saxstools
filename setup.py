from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os.path
import numpy

ext_modules = [Extension("saxstools.libsaxstools",
        [os.path.join('src', 'libsaxstools.pyx')],
        include_dirs = [numpy.get_include()],
	)]

scripts = [os.path.join('scripts', 'saxs_curve')]

package_data = {'saxstools': [os.path.join('data', '*.npy')],
                }

setup(name="saxstools",
      version='0.0.0',
      description='',
      author='Gydo C.P. van Zundert',
      author_email='g.c.p.vanzundert@uu.nl',
      packages=['saxstools'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      package_data = package_data,
      scripts=scripts,
      requires=['numpy', 'scipy', 'cython'],
    )
