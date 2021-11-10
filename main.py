import tensorflow as tf
import tensorflow.python.framework.config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class ImageClassificationTF:
    def __init__(self,path):
        self.path = path


    def visualization(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path+'/train',
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(315,315),
            batch_size=32,
        )
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")


    def test_train_val(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            height_shift_range=0.1,
            validation_split=0.2,
            rotation_range=30,
            brightness_range = (0.5,1.1),
            fill_mode='nearest')

        self.train_generator = train_datagen.flow_from_directory(
            self.path+'/train',
            target_size=(315, 315),
            batch_size=32,
            class_mode='binary',
        subset='training')

        validation_generator = train_datagen.flow_from_directory(
            self.path + '/train',
            target_size=(315, 315),
            batch_size=32,
            class_mode='binary',
            subset='validation')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_pictures = test_datagen.flow_from_directory(self.path+'/test/', target_size=(315, 315),
                                                         batch_size=32,
                                                         class_mode='binary')



    def augumented_visualization(self):
        imgs,labels = next(self.train_generator)
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(imgs, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()







data_visualization = ImageClassificationTF('hotdog__not_hotdog')
data_visualization.test_train_val()
data_visualization.visualization()
data_visualization.augumented_visualization()




