from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [Extension("saxstools/libsaxstools",
        ["src/libsaxstools.pyx"])]

scripts = ['scripts/saxs_curve']
#package_data = {'saxs-tools': ['data/*.npy', 'cl_kernels.cl']}

setup(name="saxstools",
      version='0.0.0',
      description='',
      author='Gydo C.P. van Zundert',
      author_email='g.c.p.vanzundert@uu.nl',
      packages=['saxstools'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      #package_data = package_data,
      scripts=scripts,
      requires=['numpy', 'scipy', 'cython'],
    )
