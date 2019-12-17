from setuptools import setup

# don't import CryMOS!
build_cpp = {'__file__': 'CryMOS/cpp/build.py'}
with open('CryMOS/cpp/build.py') as f:
    exec(f.read(), build_cpp)

setup(
    name='pm-mos-model',
    version='0.1',
    packages=['CryMOS', 'CryMOS.cpp'],
    url='https://github.com/michi7x7/pm-mos-model',
    license='GPL3',
    author='Michael Sieberer',
    author_email='michael.sieberer@infineon.com',
    description='A cryogenic model for the MOS transistor',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'si_prefix>=1.2'],
    setup_requires=['pybind11>=2.4'],
    ext_modules=[build_cpp['ext']],
    cmdclass={
        'build_ext': build_cpp['BuildExt'],
    }
)
