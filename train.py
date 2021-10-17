import argparse
from utils.data_maker import DataMaker


def parse_args():
    parser = argparse.ArgumentParser(description='Train n-gram')
    parser.add_argument('-n', type=int, default=3, help='n of n-gram')
    parser.add_argument('-f', type=str, default='data/train_set.txt', help='location of the training/testing file')
    parser.add_argument('-m', type=str, default='turing', help='discounting method')
    parser.add_argument('-inst', type=str, help='sentence for instant testing')
    parser.add_argument('-th', type=int, default=1, help='threshold for deleting low-frequency words')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = DataMaker(args.f)
    train_data.replace_low_frequency_word(threshold=args.th)
    train_data.total_count(max_n=args.n)
    train_data.discounting(method=args.m)
    train_data.save_bank()


if __name__ == '__main__':
    main()