import os
import json
import numpy as np

def load_bank(n: int):
    '''save n-gram bank json file as dict'''
    '''input: n: n-gram bank'''
    '''return: n_gram_bank'''
    assert os.path.exists('./n_gram_bank/%d-gram bank.json'), 'n-gram bank not trained'  # first run train.py!
    with open('./n_gram_bank/%d-gram bank.json' % n, 'r') as f:
        n_gram_bank = json.load(f)
        print(type(n_gram_bank))
        return n_gram_bank