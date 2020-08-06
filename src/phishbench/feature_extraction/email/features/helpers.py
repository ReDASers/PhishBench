from itertools import groupby

from nltk.stem import PorterStemmer


# Vocabulary richness https://swizec.com/blog/measuring-vocabulary-richness-with-python/swizec/2528
def yule(entry):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = {}
    stemmer = PorterStemmer()
    cleaned = [w.strip("0123456789!:,.?(){}[]") for w in entry.split()]
    words = filter(lambda w: len(w) > 0, cleaned)
    for w in words:
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1

    m1 = float(len(d))
    m2 = sum([len(list(g)) * (freq ** 2) for freq, g in groupby(sorted(d.values()))])
    try:
        return (m1 * m1) / (m2 - m1)
    except ZeroDivisionError:
        return 0
        # endregion
