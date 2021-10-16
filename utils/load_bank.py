import os
import json
import numpy as np

def load_bank(n: int):
    '''save n-gram bank json file as dict'''
    '''input: n: n-gram bank'''
    '''return: n_gram_bank'''
    assert os.path.exists('./n_gram_bank/%d-gram bank.json' % n), 'n-gram bank not trained'  # first run train.py!
    print('Loading %d-gram bank...' % n)
    with open('./n_gram_bank/%d-gram bank.json' % n, 'r') as f:
        n_gram_bank = json.load(f)
        return n_gram_bank