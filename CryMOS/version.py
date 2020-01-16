__version__ = None

try:  # try to get version from git
    from subprocess import run
    import re
    from os.path import dirname
    wdir = dirname(__file__)

    proc = run('git describe --dirty --tags',
               capture_output=True, shell=True, text=True, cwd=wdir)  # check=True,
    tagname = proc.stdout.strip()

    if proc.returncode != 0 or not tagname:
        proc = run('git show --format=%h -s',
                   capture_output=True, shell=True, check=True, text=True, cwd=wdir)
        h = proc.stdout.strip()
        ver = f'0.dev+git.{h}'
    elif re.fullmatch(r'v[0-9]+(\.[0-9]+){0,2}', tagname):
        ver = tagname[1:]
    else:
        m = re.fullmatch(
            r'v(?P<version>[0-9]+(\.[0-9]+){0,2})-(?P<dist>\d+)-g(?P<hash>[0-9a-f]+)(?P<dirty>-dirty)?',
            tagname)
        assert m is not None, "git describe not parsed"
        ver = f'{m.group("version")}.dev{m.group("dist")}+git.{m.group("hash")}'
        if m.group('dirty'):
            ver += '.dirty'

    __version__ = ver
except:
    pass
