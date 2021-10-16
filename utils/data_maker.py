import os
import json
import numpy as np
from utils.discounting import *

class DataMaker:
    def __init__(self, filename: str):
        '''load train/test sentences'''
        '''input: filename: location of the data to be loaded'''
        print('Loading data...')
        file = open(filename)
        self.data = file.read().split()
        self.n_gram_bank = []
    
    def single_count(self, n=1, threshold=1) -> dict:
        '''private: count n-word tokens and'''
        '''input: n: n-gram, threshold: tokens counted under the threshold are deleted'''
        '''return: n_gram dict, eg. {eat-lunch:39, I-eat:42, <s>-I:72}'''
        print('Counting %d-gram tokens...' % n)
        n_dict = {}
        data = ['<s>'] + self.data + ['</s>']
        for index in range(0, len(data) - n + 1):
            token = '-'.join(data[index: index+n])
            if n_dict.__contains__(token):
                n_dict[token] += 1
            else:
                n_dict[token] = 1.0
        len_old = len(n_dict)
        '''filter low-frequency tokens'''
        n_dict_new = {}
        for key, value in n_dict.items():
            if value > threshold:
                n_dict_new[key] = value
        n_dict_new['unk'] = 0
        len_new = len(n_dict_new)
        print('%d/%d' % (len_new, len_old))
        return n_dict_new

    def total_count(self, max_n=3, threshold=1):
        '''count 1~max_n-word tokens'''
        '''input: max_n: 1~n-gram, threshold: tokens counted under the threshold are deleted'''
        for n in range(max_n):
            self.n_gram_bank.append(self.single_count(n + 1, threshold))
    
    def discounting(self, method=''):
        '''exert discounting method on n-gram banks'''
        '''input: method: disounting method, eg. turing, gumbel'''
        if method == '':
            pass
        for n, bank in enumerate(self.n_gram_bank):
            print('Exerting %s discounting on %d-gram bank...' % (method, n + 1))
            tokens = bank.keys()
            counts = np.asarray(list(bank.values()))
            if method == 'turing':
                counts = good_turing_discount(np.asarray(counts))
            elif method == 'gumbel':
                counts = gumbel_discount(np.asarray(counts))
            '''calculate frequency'''
            frequency = counts / counts.sum()
            for m, token in enumerate(tokens):
                self.n_gram_bank[n][token] = frequency[m]
    
    def save_bank(self):
        '''save n-gram bank as json files'''
        print('Saving n-gram banks...')
        if not os.path.exists('./n_gram_bank'):
            os.makedirs('./n_gram_bank')
        for n in range(len(self.n_gram_bank)):
            n_dict = self.n_gram_bank[n]
            with open('./n_gram_bank/%d-gram bank.json' % (n + 1), 'w') as f:
                json.dump(n_dict, f)
            

