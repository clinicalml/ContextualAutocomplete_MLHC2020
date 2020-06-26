import pickle
from functools32 import lru_cache
import os

BASE_FP = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

@lru_cache(maxsize=1)
def _umls_helper():
    """Return UMLS, lookup, trie.
    """
    with open(os.path.join(BASE_FP, 'models/hpi_autocomplete/jclinic_umls.pkl'), 'rb') as h:
        return pickle.load(h)


@lru_cache(maxsize=1)
def UMLS():
    return _umls_helper()[0]


@lru_cache(maxsize=1)
def lookup():
    return _umls_helper()[1]


@lru_cache(maxsize=1)
def trie():
    return _umls_helper()[2]