import tensorflow.keras as keras
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf

BATCH_SIZE = 64


def load_train_data():
    # trdata = keras.preprocessing.image.ImageDataGenerator()
    # traindata = trdata.flow_from_directory(
    #     directory="data/train", target_size=(224, 224)
    # )

    dataset = keras.preprocessing.image_dataset_from_directory(
        "data/train/",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        validation_split=0.3,
        subset="training",
        seed=42,
    )
    return dataset


def load_validation_data():
    dataset = keras.preprocessing.image_dataset_from_directory(
        "data/train/",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        validation_split=0.3,
        subset="validation",
        seed=42,
    )
    return dataset


def load_test_data():
    # tsdata = ImageDataGenerator()
    # testdata = tsdata.flow_from_directory(directory="../test", target_size=(224, 224))

    dataset = keras.preprocessing.image_dataset_from_directory(
        "data/test1/", batch_size=BATCH_SIZE, image_size=(224, 224), labels=None
    )
    return dataset


def show_data_sample(dataset):
    class_names = dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def build_model():
    base_model = keras.applications.VGG16(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Flatten()(x)
    # x = keras.layers.Flatten()(inputs)
    # x = keras.layers.Dense(1024, activation="relu")(x)
    # x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
    #     metrics=["accuracy"],
    # )

    return model


def train_model(
    model, n_epochs, train_dataset, validation_dataset, tensorboard_callback
):
    model.fit(
        train_dataset,
        epochs=n_epochs,
        validation_data=validation_dataset,
        batch_size=16,
        callbacks=[tensorboard_callback],
    )


def main():
    train_dataset = load_train_data()
    validation_dataset = load_validation_data()
    # test_dataset = load_test_data()
    show_data_sample(train_dataset)
    show_data_sample(validation_dataset)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    # test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    model = build_model()
    tboard_callback = keras.callbacks.TensorBoard(
        log_dir=f"logs/{time()}", histogram_freq=1, profile_batch="500,520"
    )

    train_model(
        model,
        5,
        train_dataset,
        validation_dataset,
        tboard_callback,
    )


if __name__ == "__main__":
    main()
