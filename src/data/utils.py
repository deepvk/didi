def get_mode(base_name):
    if "t5" in base_name:
        return "t5"
    elif "blenderbot" in base_name:
        return "blenderbot"
    else:
        raise ValueError(f"Unsupported base name: {base_name}")


def wrap_output(iter):
    def wrapper(x):
        bos = x.bos_token
        eos = x.eos_token
        for src, trg in iter(x):
            yield bos + src + eos, bos + trg + eos

    return wrapper
