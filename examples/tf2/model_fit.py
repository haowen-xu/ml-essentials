import tensorflow.keras as keras
from tensorflow.keras import layers


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()

    train_images = (train_images / 255.0).reshape([-1, 28, 28, 1])
    test_images = (test_images / 255.0).reshape([-1, 28, 28, 1])

    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                      input_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=2,
                      activation='relu'),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=2,
                      activation='relu'),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1, verbose=2)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


