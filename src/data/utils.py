def wrap_output(iter):
    def wrapper(x):
        bos = x.bos_token
        eos = x.eos_token
        for src, trg in iter(x):
            yield bos + src + eos, bos + trg + eos

    return wrapper
