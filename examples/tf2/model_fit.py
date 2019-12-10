import tensorflow.keras as keras
from tensorflow.keras import layers

from mltk.integration import tf2


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()

    train_images = (train_images / 255.0).reshape([-1, 784])
    test_images = (test_images / 255.0).reshape([-1, 784])

    model = keras.Sequential([
        layers.Dense(500, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    tf2.model_fit(
        model, train_images, train_labels, epochs=1,
        validation_split=0.2
    )

    test_loss, test_acc = tf2.model_evaluate(model, test_images, test_labels)
