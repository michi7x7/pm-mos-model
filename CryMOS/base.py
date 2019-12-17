import numpy as np
from si_prefix import split, prefix, si_prefix_expof10
from warnings import warn

# this is the biggest float64 that can be put in exp()
F64_LMAX = np.log(np.finfo(np.float).max)


def si_scale(x: np.ndarray, pref=None):
    if pref:
        e10 = si_prefix_expof10(pref)
    else:
        mx = np.amax(np.abs(x))
        _, e10 = split(mx)
        pref = prefix(e10)

    return x / 10**e10, pref


def sq_scale(x, unit="m"):
    v, pref = si_scale(np.sqrt(x))
    return f"{v**2:.2f} ({pref}{unit})^2"


def arg_dict(arg, **kwargs):
    if type(arg) == dict:
        return {**kwargs, **arg}
    else:
        return kwargs


def ndarray_by_ref(arg, shape, name="arg"):
    """return an intermediate result by reference (assign to ndarray)"""
    if isinstance(arg, np.ndarray):
        arg.reshape(shape)
    elif arg is None:
        arg = np.ndarray(shape)
    else:
        raise TypeError(f"{name} must be ndarray or None")
    return arg


def cond(f, default, arg=None):
    if arg is False:
        return
    elif arg is None or arg is True:
        return f(default)
    else:
        return f(arg)


class FitMe:
    @classmethod
    def extract(cls, p, *args):
        """ extract FitMe-values
            p is a list of values, every `FitMe` in `args` will be replaced by consecutive `p` values.
            >>> tuple(FitMe.extract([9, 13], 1, FitMe(2), 3, FitMe(4), 5)) == (1, 9, 3, 13, 5)
        """
        p = iter(p)
        for a in args:
            if isinstance(a, cls):
                yield next(p)
            else:
                yield a

    @classmethod
    def kextract(cls, p, **kargs):
        """ extract FitMe-values
            p is a list of values, every `FitMe` in `args` will be replaced by consecutive `p` values.
            >>> FitMe.kextract([9, 13], a=1, b=FitMe(2), c=3, d=FitMe(4), e=5) == {'a':1, 'b':9, 'c':3, 'd':13, 'e':5}
        """
        k = kargs.keys()
        v = cls.extract(p, *kargs.values())
        return dict(zip(k, v))

    @classmethod
    def kextract_fits(cls, p, **kargs):
        """ extract FitMe-values
            p is a list of values, every `FitMe` in `args` will be returned.
            >>> FitMe.kextract_fits([9, 13], a=1, b=FitMe(2), c=3, d=FitMe(4), e=5) == {'b': 9, 'd': 13}
        """
        p = iter(p)
        return {k: next(p) for k, v in kargs.items() if isinstance(v, cls)}

    @classmethod
    def param_dict(cls, **kargs):
        """ extract start parameters """
        return {k: (v.start if isinstance(v, cls) else v) for k, v in kargs.items()}

    @classmethod
    def start_list(cls, *args):
        """ start values from a list of arguments
            >>> FitMe.start_list(1, FitMe(2), 3, FitMe(4), 5) == [2,4]
            """
        return [x.start for x in args if isinstance(x, cls)]

    @classmethod
    def bound_list(cls, *args):
        a = [x.lb for x in args if isinstance(x, cls)]
        b = [x.ub for x in args if isinstance(x, cls)]
        return a, b

    def __init__(self, start, lb=-np.inf, ub=np.inf):
        self.start = start
        self.lb = lb
        self.ub = ub

    def __bool__(self):
        return bool(self.start)


def writeable_property(default_fun):
    """ function decorator that allows overwriting default properties """
    pname = '_' + default_fun.__name__

    @property
    def pfun(self):
        attr = getattr(self, pname, None)
        if attr is None:
            return default_fun(self)

        if callable(attr):
            return attr(self)
        else:
            return attr

    @pfun.setter
    def pfun(self, newval):
        return setattr(self, pname, newval)

    pfun.__doc__ = default_fun.__doc__
    return pfun


class MosModelMeta(type):
    # TODO: I don't really like this
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        if not hasattr(x, 'mdl_cls'):
            inst = x()
            x.param_default = {k: getattr(inst, k) for k in x.params}
        return x


class MosModelBase(metaclass=MosModelMeta):
    params = ('w', 'l', 'temp')
    pandas_default = ('temp',)

    """ just very basic parameters """
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Invalid attribute {k}!")
            try:
                setattr(self, k, v)
            except AttributeError as e:
                raise AttributeError(f"can't set attribute {k}") from e

    def __init__(self, **kwargs):
        self.w = None
        self.l = None

        self.temp = 300.
        self.update_params(**kwargs)
    
    @property
    def area(self):
        if self.l and self.w:
            return self.l * self.w
        else:
            return None

    @classmethod
    def from_mdl(cls, oth, **kwargs):
        return oth.copy_mdl(cls=cls, **kwargs)

    def copy_mdl(self, cls=None, **kwargs):
        keys = self.params

        if cls:
            new = cls()
        elif hasattr(self, 'mdl_cls'):
            new = self.mdl_cls()
        else:
            new = type(self)()

        for k in keys:
            try:
                v = getattr(self, k)
            except AttributeError:
                warn(f"old model has no key '{k}', leaving default")  # {cls.param_default[k]}
            else:
                setattr(new, k, v)

        new.update_params(**kwargs)
        return new

    def param_sweep(self, **kwargs):
        return ModelSweep(self, **kwargs)

    def param_dict(self):
        return {k: getattr(self, k) for k in self.params}

    def fit_arb(self, fun,  x, y, method=None, sigma=None, write_parameters=True, **kwargs):
        """ Fits model parameters to an arbitrary callback function """
        from scipy.optimize import curve_fit

        x = np.asarray_chkfinite(x)
        y = np.asarray_chkfinite(y)
        assert x.ndim == y.ndim == 1
        assert x.shape == y.shape

        # calculate initial variables
        mdl = self.copy_mdl(**FitMe.param_dict(**kwargs))

        def callbk(xf, *p):
            mdl.update_params(**FitMe.kextract(p, **kwargs))
            return fun(mdl, xf)

        popt, pcov = curve_fit(callbk, x, y, p0=FitMe.start_list(*kwargs.values()),
                               bounds=FitMe.bound_list(*kwargs.values()), method=method, sigma=sigma,absolute_sigma= True)
        # TODO: check quality of fit!
        # TODO: handle warnings
        # TODO: logging

        params = FitMe.kextract(popt, **kwargs)
        if write_parameters:
            self.update_params(**params)

        return params, popt, pcov

    def __str__(self):
        return str("\n".join(f"{k}:\t{getattr(self, k)}" for k in self.params))


def unique_list(a):
    b = []
    list(b.append(x) for x in a if x not in b)
    return b


class ModelCollection:
    def __init__(self, models: list = None):
        self._models = models or []

    @property
    def models(self):
        """ returms a list of all swept models """
        return self._models

    @property
    def default_keys(self):
        try:
            return self.models[0].pandas_default
        except:
            return ()

    def append(self, mdl):
        self._models.append(mdl)

    def __iter__(self):
        """ can be used in a for loop """
        return iter(self.models)

    def __getitem__(self, item):
        return self.models[item]

    def __getattr__(self, item):
        def sweep_fun(mdl):
            return getattr(mdl, item)

        res = np.vectorize(sweep_fun)(self.models)

        if not callable(res[0]):
            return res

        # we have an array of callibles, make dummy caller
        def caller(*args, **kwargs):
            def callx(f):
                return f(*args, **kwargs)
            try:
                return np.vectorize(callx)(res)
            except ValueError:
                # our function returns an array
                return np.vectorize(callx, signature='()->(m)')(res)

        return caller

    def pandas(self, *retkeys):
        from pandas import DataFrame
        if len(retkeys) == 0:
            retkeys = self.default_keys
        elif len(retkeys) == 1 and type(retkeys[0]) is not str:
            retkeys = tuple(retkeys[0])

        def sweep_fun(mdl):
            return tuple(getattr(mdl, i) for i in retkeys)

        res = np.vectorize(sweep_fun, otypes=[tuple])(self.models)
        return DataFrame(list(res), columns=retkeys)


class ModelSweep(ModelCollection):
    def __init__(self, mdl, grid=False, **sweeps):
        super().__init__()
        self.mdl: MosModelBase = mdl

        if grid is True:
            v = np.meshgrid(*sweeps.values())
            self.sweeps = dict(zip(sweeps.keys(), v))
        else:
            self.sweeps = sweeps

    def __str__(self):
        return str(self.pandas())

    @property
    def models(self):
        keys = self.sweeps.keys()

        def sweep_fun(*p):
            par = dict(zip(keys, p))
            return self.mdl.copy_mdl(**par)

        return np.vectorize(sweep_fun)(*self.sweeps.values())

    @property
    def default_keys(self):
        keys = tuple(self.sweeps.keys()) + super().default_keys
        return unique_list(keys)
