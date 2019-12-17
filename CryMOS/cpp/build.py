from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools, distutils


class get_pybind_include:
    """ Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def get_boost_include(builddir='build'):
    import os

    root = None

    if 'BOOST_ROOT' in os.environ:
        root = os.environ['BOOST_ROOT']
    else:
        path = [builddir, '.', '~', __file__ + '../../build']

        for p in path:
            root = os.path.expanduser(p + '/boost_1_72_0')
            if os.path.exists(root):
                break

    if root is None or not os.path.exists(root + '/boost'):
        from warnings import warn
        warn("boost libraries could not be found, you might want to set `BOOST_ROOT`")

    return root


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++14')
            opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


class BuildImportCppExt(BuildExt):
    """ copy ext to source directory (not used right now) """
    def copy_extensions_to_source(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src_filename = os.path.join(self.build_lib, filename)
            dest = os.path.dirname(__file__)  # copy ext to this folder
            dest_filename = os.path.join(dest, os.path.basename(filename))

            distutils.file_util.copy_file(
                src_filename, dest_filename,
                verbose=self.verbose, dry_run=self.dry_run
            )


class BuildAndImportCppExt(BuildExt):
    """ copy ext to source directory (not used right now) """
    def copy_extensions_to_source(self):
        import importlib.util

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src_filename = os.path.join(self.build_lib, filename)

            import importlib.util
            spec = importlib.util.spec_from_file_location(ext.name, src_filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ext.mod = module


cppfile = os.path.dirname(__file__) + '/mod.cpp'

ext = Extension(
    'CryMOS.cpp.mod',
    [cppfile],
    include_dirs=[
        # Path to pybind11 headers
        get_pybind_include(),
        get_pybind_include(user=True),
        get_boost_include()
    ],
    language='c++',
    optional=True
)


def import_inplace():
    import tempfile
    import shutil
    build_path = tempfile.mkdtemp()

    args = ['build_ext', '--inplace']
    args.append('--build-temp=' + build_path)
    args.append('--build-lib=' + build_path)

    setuptools_args = dict(
        name="mod",
        ext_modules=[ext],
        script_args=args,
        cmdclass={
            'build_ext': BuildAndImportCppExt
        }
    )

    setuptools.setup(**setuptools_args)

    # remove temporary files again
    import atexit

    @atexit.register
    def del_mod():
        del sys.modules[ext.name]
        try:
            shutil.rmtree(build_path)
        except:
            from warnings import warn
            warn(f"please delete {build_path}")
            pass

    return ext.mod
