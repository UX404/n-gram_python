import os
import json
import numpy as np
from collections import Counter
from utils.discounting import *

class DataMaker:
    def __init__(self, filename: str):
        '''load train/test sentences'''
        '''input: filename: location of the data to be loaded'''
        print('Loading data...')
        file = open(filename)
        self.data = file.read().split()
        self.n_gram_bank = []
    
    def __len__(self):
        '''return: length of the data'''
        return len(self.data)

    def replace_low_frequency_word(self, threshold=1):
        '''remove low-frequency words'''
        '''threshold: words counted under the threshold are replaced by <unk>'''
        counts = Counter(self.data)
        for n in range(len(self.data)):
            if counts[self.data[n]] <= threshold:
                self.data[n] = '<unk>'
    
    def replace_unseen_word(self, uni_bank: dict):
        '''replace unseen words in the training word bank'''
        '''input: uni_bank: 1-gram word bank'''
        uni_bank = set(uni_bank.keys())
        for n in range(len(self.data)):
            if not self.data[n] in uni_bank:
                self.data[n] = '<unk>'
    
    def single_count(self, n=1) -> dict:
        '''private: count n-word tokens and'''
        '''input: n: n-gram'''
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
        if not n_dict.__contains__('-'.join(['<unk>'] * n)):
            n_dict['-'.join(['<unk>'] * n)] = 1e-3
        print(len(n_dict))
        return n_dict

    def total_count(self, max_n=3):
        '''count 1~max_n-word tokens'''
        '''input: max_n: 1~n-gram, threshold: tokens counted under the threshold are deleted'''
        for n in range(max_n):
            self.n_gram_bank.append(self.single_count(n + 1))
    
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
            for m, token in enumerate(tokens):
                self.n_gram_bank[n][token] = counts[m]
    
    def save_bank(self):
        '''save n-gram bank as json files'''
        print('Saving n-gram banks...')
        if not os.path.exists('./n_gram_bank'):
            os.makedirs('./n_gram_bank')
        for n in range(len(self.n_gram_bank)):
            n_dict = self.n_gram_bank[n]
            with open('./n_gram_bank/%d-gram bank.json' % (n + 1), 'w') as f:
                json.dump(n_dict, f)
    
    def calculate_ppl(self, n: int, v_bank: dict, y_bank: dict) -> float: 
        '''calculate PPL'''
        '''intput: n: n-gram, v_bank: n-gram frequency bank, y_bank: n-1-gram frequency bank'''
        '''return: PPL'''
        print('Calculating PPL...')
        ppl = 1
        k = self.__len__()
        data = ['<s>'] + self.data + ['</s>']
        for index in range(n - 1, len(data)):
            v_token = '-'.join(data[index - n + 1: index + 1])
            y_token = '-'.join(data[index - n + 1: index])
            if not v_bank.__contains__(v_token):
                v_token = '-'.join(['<unk>'] * n)
            if not y_bank.__contains__(y_token):
                y_token = '-'.join(['<unk>'] * (n - 1))
            ppl *= np.power(1 / (v_bank[v_token] / y_bank[y_token]), 1 / k)
            # print(ppl, v_token, y_token)
        return ppl