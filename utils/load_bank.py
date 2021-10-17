import os
import json
import numpy as np

def load_bank(max_n: int):
    '''save n-gram bank json file as dict'''
    '''input: max_n: n-gram bank'''
    '''return: 1~n_gram_bank'''
    assert os.path.exists('./n_gram_bank/%d-gram bank.json' % max_n), 'n-gram bank not trained'  # first run train.py!
    print('Loading 1~%d-gram bank...' % max_n)
    bank = {'': 1}
    for n in range(1, max_n + 1):
        with open('./n_gram_bank/%d-gram bank.json' % n, 'r') as f:
            n_gram_bank = json.load(f)
            bank.update(n_gram_bank)
    return bank