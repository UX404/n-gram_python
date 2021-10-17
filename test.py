import argparse
from utils.data_maker import DataMaker
from utils.load_bank import load_bank


def parse_args():
    parser = argparse.ArgumentParser(description='Train n-gram')
    parser.add_argument('-n', type=int, default=3, help='n of n-gram')
    parser.add_argument('-f', type=str, default='data/train_set.txt', help='location of the training/testing file')
    parser.add_argument('-inst', type=str, default='', help='sentence for instant testing')
    return parser.parse_args()


def main():
    args = parse_args()
    # handle instant input
    if args.inst != '':
        inst = args.inst.split('-')
        inst = ' '.join(inst)
        with open('./data/temp.txt', 'w') as f:
            f.write(inst)
        args.f = './data/temp.txt'

    uni_bank = load_bank(max_n=1)
    bank = load_bank(max_n=args.n)
    test_data = DataMaker(args.f)
    test_data.replace_unseen_word(uni_bank)
    ppl = test_data.calculate_ppl(n=args.n, bank=bank)
    print('PPL = %.5f' % ppl)


if __name__ == '__main__':
    main()