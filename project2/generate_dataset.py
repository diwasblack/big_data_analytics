import joblib
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


HEIGHT = 200
WIDTH = 200
CHANNELS = 3
BATCH_SIZE = 32
CLASSES = 10
SAMPLES = 272  * 3


def generate_dataset(dataset_path, filename):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        horizontal_flip=True
    )

    data_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        # save_to_dir="processed"
    )

    # Construct a numpy array to store the data
    features = np.zeros(shape=(SAMPLES, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
    labels = np.zeros(shape=(SAMPLES, CLASSES), dtype=np.float32)

    start = 0
    while (start < SAMPLES):
        input_batch, input_label = data_generator.next()

        diff = input_batch.shape[0]

        features[start: start+diff] = input_batch
        labels[start: start+diff] = input_label

        start += diff

    joblib.dump((features, labels), filename, compress=True)


if __name__ == "__main__":
    dataset_path = "10-monkey-species/validation"
    generate_dataset(dataset_path, "dataset_monkey_test.joblib")
