import re
import pandas as pd


def ngrams(string, n=3, reg=r"[,-./]|\sBD"):
    """Generate a full list of ngrams from a list of strings
    :param string: List of strings to generate ngrams from.
    :type string: list (of strings)
    :param n: Maximum length of the n-gram. Defaults to 3.
    :type n: int
    :param reg: regular expression
    :type reg: str
    :raises AssertionError: If you pass in a list that contains datatypes other than `string`, you're in trouble!
    :return: Returns list of ngrams generated from the input string.
    :rtype: list
    """

    # Assert string type
    assert isinstance(string, str), "String not passed in!"

    # Remove Punctuation from the string
    string = re.sub(reg, r" ", string)

    # Generate zip of ngrams (n defined in function argument)
    n_grams = zip(*[string[i:] for i in range(n)])

    # Return ngram list
    return ["".join(ngram) for ngram in n_grams]


if __name__ == '__main__':
    print(ngrams(pd.read_csv('data/new_items_test.csv').loc[0,'name_1'],
                 n=3, reg=r"[,-./]|\sBD"))
