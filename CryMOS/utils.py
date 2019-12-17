
class SiMemo:
    def __init__(self, fun):
        self.t = None
        self.memo = None
        self.fun = fun

    def __call__(self, t):
        if t is self.t:
            return self.memo

        self.t = t
        self.memo = self.fun(t)
        return self.memo
