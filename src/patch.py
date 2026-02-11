import numpy as np

# Monkey patch numpy.asarray to handle 'copy' argument expected by newer JAX
# but seemingly missing in the installed numpy version.

_orig_asarray = np.asarray


def _patched_asarray(a, dtype=None, order=None, *, copy=None, **kwargs):
    # JAX passes copy=True/False/None.
    # Older numpy.asarray doesn't take 'copy'.
    # np.array takes 'copy'.
    if copy is True:
        return np.array(a, dtype=dtype, order=order, copy=True, **kwargs)
    elif copy is False:
        # np.asarray does not copy by default if requirements met
        return _orig_asarray(a, dtype=dtype, order=order, **kwargs)
    else:
        return _orig_asarray(a, dtype=dtype, order=order, **kwargs)


np.asarray = _patched_asarray
