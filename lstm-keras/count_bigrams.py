from collections import Counter, defaultdict
from itertools import tee

#function from 'recipes section' in standard documentation itertools page
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def bigrams(text):
    c = Counter()
    c.update(pairwise(text.split()))
    return c.items()
