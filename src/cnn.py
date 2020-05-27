from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np

from constants import START_LEARNING_RATE, MIN_LEARNING_RATE, BATCH_SIZE, EPOCHS, PATH_TO_RESULTS


class CNN():

    def __init__(self):
        self.cnn = None
        self.results = None

    def init_model(self):
        self.cnn = Sequential([
            Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.5),
            Conv2D(32, (5, 5), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.5),
            Conv2D(32, (3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10)
        ])

    def compile_model(self):
        opt = Adam(learning_rate=START_LEARNING_RATE)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.cnn.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    def train_model(self, train_data_preprocessor):
        digits = train_data_preprocessor.get_digits()
        labels = train_data_preprocessor.get_labels()

        reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.5,
                                      patience=3,
                                      verbose=1,
                                      min_lr=MIN_LEARNING_RATE)

        self.cnn.fit(digits,
                     labels,
                     batch_size=BATCH_SIZE,
                     epochs=EPOCHS,
                     callbacks=[reduce_lr])

    def evaluate_model(self, test_data_preprocessor):
        test_digits = test_data_preprocessor.get_digits()
        results = self.cnn.predict(test_digits)
        self.store_results(results)

    @staticmethod
    def store_results(results):
        results_df = pd.DataFrame(columns=['ImageId', 'Label'])
        results_df['Label'] = np.argmax(results, axis=1)
        results_df['ImageId'] = np.arange(1, len(results)+1)
        results_df.to_csv(PATH_TO_RESULTS, index=False)