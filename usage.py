from classes import ImageClassificationTF as imcTF
from classes import ImageClassificationPT
from tensorflow.python.client import device_lib


# Tensorflow/Keras
def tensorflow_classifier():
    image_class_keras = imcTF('hotdog__not_hotdog', image_size=(315, 315))  # Fitting samples into a class
    train_test_val = image_class_keras.test_train_val()  # Data split
    image_class_keras.augumented_visualization()  # Augmented data visualization
    vgg16_f = image_class_keras.pretrained_model(model='VGG16', unfreeze=0)  # Pretrained VGG16 with freezed layers
    vgg19_f = image_class_keras.pretrained_model(model='VGG19', unfreeze=0)  # Pretrained VGG19 with freezed layers
    vgg16_u = image_class_keras.pretrained_model(model='VGG16', unfreeze=1)  # Pretrained VGG16 with unfreezed layers
    vgg19_u = image_class_keras.pretrained_model(model='VGG19', unfreeze=1)  # Pretrained VGG19 with unfreezed layers
    models = [vgg16_f, vgg19_f, vgg16_u, vgg19_u]
    for model in models:
        image_class_keras.training_process(model, 100, *train_test_val)


# Pytorch
def torch_classifier():
    imageclass = ImageClassificationPT('hotdog__not_hotdog', (315, 315))
    train_val_test = imageclass.load_split_train_test(0.2)
    imageclass.visualize_classification(train_val_test[0])

    # Resnet50 with frozen layers
    imageclass.pretrained_model('resnet', 0, *imageclass.load_split_train_test(0.2), 50)
    # Densenet with frozen layers
    imageclass.pretrained_model('densenet', 0, *imageclass.load_split_train_test(0.2), 50)
    # Resnet50 with unfreezed layers
    imageclass.pretrained_model('resnet', 1, *imageclass.load_split_train_test(0.2), 50)
    # Densenet with unfreezed layers
    imageclass.pretrained_model('densenet', 1, *imageclass.load_split_train_test(0.2), 50)


if __name__ == "__main__":
    tensorflow_classifier()
    # torch_classifier()
