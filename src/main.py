from data_preprocessing import DataPreprocessor
from cnn import CNN


def main():
    train_data_preprocessor = DataPreprocessor(dataset='train')
    test_data_preprocessor = DataPreprocessor(dataset='test')

    train_data_preprocessor.preprocess()
    test_data_preprocessor.preprocess()

    cnn = CNN()
    cnn.init_model()
    cnn.compile_model()
    cnn.train_model(train_data_preprocessor)
    cnn.evaluate_model(test_data_preprocessor)
    print('done')

if __name__ == '__main__':
    main()