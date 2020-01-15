from setuptools import setup
import distutils.cmd

# don't import CryMOS!
build_cpp = {'__file__': 'CryMOS/cpp/build.py'}
with open('CryMOS/cpp/build.py') as f:
    exec(f.read(), build_cpp)


class DwnlBoostCommand(distutils.cmd.Command):
    """A custom command to run Pylint on all Python source files."""

    description = 'Download boost to build directory'
    user_options = [
        # The format is (long option, short option, description).
        ('version=', None, 'Boost version (def. 1.72.0)'),
        ('hash=', None, 'zip-file hash (8c20440aaba21dd963c0f7149517445f50c62ce4eb689df2b5544cc89e6e621e)')
    ]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.version = '1.72.0'
        self.hash = '8c20440aaba21dd963c0f7149517445f50c62ce4eb689df2b5544cc89e6e621e'

    def finalize_options(self):
        """Post-process options."""
        self.fname = 'boost_' + self.version.replace('.', '_')

    def run(self):
        """Run command."""

        from tqdm import tqdm
        from zipfile import ZipFile
        import os.path

        print("Downloading boost")
        contents = self.download_boost_archive()

        print("Preparing to extract")
        zipfile = ZipFile(contents)

        print("Extracting")
        if not os.path.isdir("build"):
            os.mkdir("build")

        filelist = zipfile.namelist()
        filelist = list(f for f in filelist if f.startswith(self.fname + '/boost/'))

        for file in tqdm(iterable=filelist, total=len(filelist), unit='files'):
            zipfile.extract(member=file, path="build")

    def download_boost_archive(self):
        from tqdm import tqdm
        import requests
        from io import BytesIO
        from hashlib import sha256

        url = f"https://dl.bintray.com/boostorg/release/1.72.0/source/{self.fname}.zip"

        with requests.get(url, stream=True, timeout=3) as r:
            r.raise_for_status()
            contents = BytesIO()
            total_len = int(r.headers['Content-Length'])

            pbar = tqdm(total=total_len, unit='B', unit_scale=True)
            for chunk in r.iter_content(chunk_size=8192):
                contents.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

        buf = contents.getbuffer()
        assert len(buf) == total_len

        if self.hash is not None:
            sh = sha256()
            sh.update(buf)
            assert self.hash == sh.hexdigest()

        return contents


setup(
    name='pm-mos-model',
    version='0.1',
    packages=['CryMOS', 'CryMOS.cpp'],
    url='https://github.com/michi7x7/pm-mos-model',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License'],
    author='Michael Sieberer',
    author_email='michael.sieberer@infineon.com',
    description='A cryogenic model for the MOS transistor',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'si_prefix>=1.2', 'fdint>=2.0'],
    setup_requires=['pybind11>=2.4'],
    ext_modules=[build_cpp['ext']],
    cmdclass={
        'download_boost': DwnlBoostCommand,
        'build_ext': build_cpp['BuildExt'],
    }
)
