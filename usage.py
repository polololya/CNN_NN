from classes import ImageClassificationTF as imcTF
from classes import ImageClassificationPT

# Tensorflow/Keras
image_class_keras = imcTF('hotdog__not_hotdog', image_size=(315, 315))#Fitting samples into a class
train_test_val = image_class_keras.test_train_val()#Data split
image_class_keras.augumented_visualization()#Augmented data visualization

#Define models
self_defined = image_class_keras.customized_model()
vgg16_f = hot_dog.pretrained_model(model='VGG16', unfreeze=0)#Pretrained VGG16 with freezed layers
vgg19_f = hot_dog.pretrained_model(model='VGG19', unfreeze=0)#Pretrained VGG19 with freezed layers
vgg16_u = hot_dog.pretrained_model(model='VGG16', unfreeze=1)#Pretrained VGG16 with unfreezed layers
vgg19_u = hot_dog.pretrained_model(model='VGG19', unfreeze=1)#Pretrained VGG19 with unfreezed layers
keras_models = [self_defined, vgg16_f, vgg19_f, vgg16_u, vgg19_u ]

#Train all 5 models and evaluate them
for model in keras_models:
    image_class_keras.training_process(model, 50, *train_test_val)  # Training


#image_class_keras.training_process(self_defined, 50, *train_test_val)#Training


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
train_val_test = imageclass.load_split_train_test(0.2)
imageclass.visualize_classification(train_val_test[0])

#Resnet50 with frozen layers
#imageclass.pretrained_model('resnet', 0, *imageclass.load_split_train_test(0.2), 50)
#Densenet with frozen layers
#imageclass.pretrained_model('densenet', 0, *imageclass.load_split_train_test(0.2), 50)
#Resnet50 with unfreezed layers
#imageclass.pretrained_model('resnet', 1, *imageclass.load_split_train_test(0.2), 50)
##Densenet with unfreezed layers
#imageclass.pretrained_model('densenet', 1, *imageclass.load_split_train_test(0.2), 50)
