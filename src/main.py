from data_preprocessing import get_train_data, get_test_data
from cnn import CNN


def main():
    cnn = CNN()
    cnn.init_model()
    cnn.compile_model()
    digits, labels = get_train_data()
    cnn.train_model(digits, labels)

    test_digits = get_test_data()
    cnn.evaluate_model(test_digits)
    cnn.store_results()
    print('done')

if __name__ == '__main__':
    main()