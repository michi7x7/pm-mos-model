__version__ = None

try:  # try to get version from git
    from subprocess import run
    from os.path import dirname
    dir = dirname(__file__)

    proc = run('git tag -l --sort=-v:refname',
               capture_output=True, shell=True, check=True, text=True, cwd=dir)
    tags = [t for t in proc.stdout.split() if t.startswith('v')]
    ver = tags[0].replace('v', '')

    proc = run('git log --abbrev-commit --format=%h -1',
               capture_output=True, shell=True, check=True, text=True, cwd=dir)
    hash = proc.stdout.strip()

    __version__ = ver + '.dev0+git.' + hash
except:
    pass
