from utils.data_maker import DataMaker
from utils.load_bank import load_bank

def main():
    uni_bank = load_bank(n=1)
    v_bank = load_bank(n=2)
    y_bank = load_bank(n=1)
    test_data = DataMaker('data/temp.txt')
    test_data.replace_unseen_word(uni_bank)
    ppl = test_data.calculate_ppl(n=2, v_bank=v_bank, y_bank=y_bank)
    print('PPL = %.5f' % ppl)


if __name__ == '__main__':
    main()