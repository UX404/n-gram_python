from utils.data_maker import DataMaker

def main():
    train_data = DataMaker('data/train_set.txt')
    train_data.total_count(max_n=3, threshold=1)
    train_data.discounting(method='gumbel')
    train_data.save_bank()


if __name__ == '__main__':
    main()