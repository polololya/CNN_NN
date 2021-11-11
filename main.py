import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense

class ImageClassificationTF:
    def __init__(self, path):
        self.path = path


    def visualization(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path + '/train',
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(315, 315),
            batch_size=32,
        )
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
                plt.suptitle('Dataset visualization with labels')

            plt.savefig('results/dataset_visualization.jpg')

    def test_train_val(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            height_shift_range=0.1,
            validation_split=0.2,
            rotation_range=30,
            brightness_range=(0.5, 1.1),
            fill_mode='nearest')

        self.train_generator = train_datagen.flow_from_directory(
            self.path + '/train',
            target_size=(315, 315),
            batch_size=32,
            class_mode='binary',
            subset='training')

        self.validation_generator = train_datagen.flow_from_directory(
            self.path + '/train',
            target_size=(315, 315),
            batch_size=32,
            class_mode='binary',
            subset='validation')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_pictures = test_datagen.flow_from_directory(self.path + '/test/', target_size=(315, 315),
                                                              batch_size=32,
                                                              class_mode='binary')
        return self.train_generator, self.validation_generator, self.test_pictures

    def augumented_visualization(self):
        imgs, labels = next(self.train_generator)
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(imgs, axes):
            ax.imshow(img)
            ax.axis('off')
            plt.suptitle('Examples of data augmentation')
        plt.tight_layout()
        plt.savefig('results/augmentation_examples.png')
        plt.show()

    def customized_model(self,image_size,num_classes):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),  # Random horizontal and verical picture change
                layers.RandomRotation(0.1),  # Random rotation of the picture - 10 degrees
                layers.RandomContrast(0.003),  # Adjust contrast for each picture independantly
                layers.RandomHeight(0.3),  # Random change height of the picture
            ]
        )

        inputs = keras.Input(shape=image_size + (3,))
        # Image augmentation block
        x = data_augmentation(inputs)
        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside residual
        for self.size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(self.size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(self.size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
            # Project residual
            residual = layers.Conv2D(self.size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        customized_model = keras.Model(inputs, outputs)

        return customized_model


    def training_process(self, model, epochs, train_ds, val_ds, test_set):
        model.summary()  # model layers+info
        # keras.utils.plot_model(model, show_shapes=True)#visualization of the model

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),  # most popular optimizer at the moment
            loss="binary_crossentropy",  # loss for classification problem
            metrics=["accuracy"],  # required metric
        )

        # Fit model with data
        history = model.fit(train_ds, epochs=epochs,
                            # callbacks=callbacks,
                            validation_data=val_ds,
                            )
        # Model results visualization

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('results/accuracy.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('results/loss.png')
        score = model.evaluate(test_set, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def pretrained_model(self,model,unfreeze):
        if model == 'VGG16':
            model = VGG16(include_top=False, input_shape=image_size + (3,), weights='imagenet')

        elif model == 'VGG19':
            model = VGG19(include_top=False, input_shape=image_size + (3,), weights='imagenet')

        customized_model = Sequential()

        # Adding first layer of default model
        customized_model.add(model.layers[0])

        # Add remaining layers of default model
        for layer in model.layers[1:-1]:
            customized_model.add(layer)

        # Add global average polling to be in line with dimensionality
        # customized_model.add(layers.GlobalAveragePooling2D())

        # Set pretrained layers of the model to be not trainable
        if unfreeze == 'yes':
            customized_model.add(layers.GlobalAveragePooling2D())
            for layer in customized_model.layers[:]:
                layer.trainable = True



        else:
            # Add global average polling to be in line with dimensionality
            customized_model.add(layers.GlobalAveragePooling2D())
            for layer in customized_model.layers[4:]:
                layer.trainable = False

        # Add last layer with sigmoid activation function since we have 2 classes
        customized_model.add(Dense(units=1, activation='sigmoid'))

        return self.customized_model

