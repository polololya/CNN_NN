from classes import ImageClassificationTF as imcTF
import tensorflow as tf
from classes import ImageClassificationPT

# Tensorflow/Keras parth
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# hot_dog = imcTF('hotdog__not_hotdog', image_size=(315, 315))
# train_test_val = hot_dog.test_train_val()#Data split
# hot_dog.visualization()#Part of data visualization
# hot_dog.augumented_visualization()#Augmented data visualization

# Baseline model(own model)
# self_defined = hot_dog.customized_model((315, 315), 2)
# hot_dog.training_process(self_defined, 50, *train_test_val)#Training

# VGG16
# vgg16 = hot_dog.pretrained_model(model='VGG16', unfreeze=0)#Pretrained VGG16 with freezed layers
# hot_dog.training_process(vgg16, 50, *train_test_val)#Training

# VGG19
# vgg19 = hot_dog.pretrained_model(model='VGG19', unfreeze=0)#Pretrained VGG19 with freezed layers
# hot_dog.training_process(vgg19, 50, *train_test_val)


# VGG16 unfreezed layers
# vgg16 = hot_dog.pretrained_model(model='VGG16', unfreeze=1)#Pretrained VGG16 with unfreezed layers
# hot_dog.training_process(vgg16, 50, *train_test_val)#Training

# VGG19 unfreezed layers
# vgg19 = hot_dog.pretrained_model(model='VGG19', unfreeze=1)#Pretrained VGG19 with unfreezed layers
# hot_dog.training_process(vgg19, 50, *train_test_val)

# Pytorch
imageclass = ImageClassificationPT('hotdog__not_hotdog', (315, 315))
imageclass.pretrained_model('densenet', 1, *imageclass.load_split_train_test(0.2))
