import numpy as np

def good_turing_discount(token: np.array, count: np.array, threshold=2):
    '''input: threshold: discounting on tokens with counts under threshold'''
    sort_index = np.argsort(count)
    token = token[sort_index]
    count = count[sort_index]
    n1 = 0
    while count[n1] < threshold:
        n2 = n1
        while count[n2] == count[n1]:  # count[n1: n2] has lower counts
            n2 += 1
        n3 = n2
        while count[n3] == count[n2]:  # count[n2: n3] has higher counts
            n3 += 1
        count[n1: n2] = (count[n1] + 1) * (n3 - n2) / (n2 - n1)  # distribute counts from count[n2: n3] to count[n1: n2]
        n1 = n3
    print('%d/%d discounted' % (n1, len(count)))
    return token, count

def gumbel_discount(count: np.array, tau=2):
    '''gumbel-softmax: Jang E, Gu S, Poole B. Categorical reparameterization with gumbel-softmax[J]. arXiv preprint arXiv:1611.01144, 2016.'''
    gumbel = np.random.random(len(count)) / 10
    count = np.exp(np.log(count + gumbel) / tau) / np.exp(np.log(count+ gumbel) / tau).sum() * count.sum()
    return count