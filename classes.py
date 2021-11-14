#Tensorflow/Keras related imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from keras_lr_finder import LRFinder
from torch.utils.data.sampler import SubsetRandomSampler
#Torch related imports
import torchvision
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
from torch_lr_finder import LRFinder as TLRFinder
from torchvision.utils import make_grid

'''
Here is the implementation of VGG16, VGG19 - with Keras and
Resnet50 and Densenet161 with Pytorch. Layers of base model were set to non trainable and then set as trainable to compare the performance of the model.
There are some visualization methods there as well. Samples visualization was performed only with keras, to omit repetitions, yet augmentation visualization was performed using both keras and torch libraries.
Optimal learning rate for all models was implemented to determine visually and to be set on user choice.(Leslie N. Smith method - https://arxiv.org/abs/1506.01186)
All visualizations were stored to the corresponding folder(*project_folder/results/**library)
*location of all of the files for the project
**library which you are using - keras/torch
'''
class ImageClassificationTF:
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size

    def visualization(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path + '/train',
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=self.image_size,
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

            plt.savefig('results/keras/dataset_visualization.jpg')

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
            batch_size=64,
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
        plt.savefig('results/keras/augmentation_examples.png')
        plt.show()

    def customized_model(self,):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),  # Random horizontal and verical picture change
                layers.RandomRotation(0.1),  # Random rotation of the picture - 10 degrees
                layers.RandomContrast(0.003),  # Adjust contrast for each picture independantly
                layers.RandomHeight(0.3),  # Random change height of the picture
            ]
        )

        inputs = keras.Input(shape=self.image_size + (3,))
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
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        customized_model = keras.Model(inputs, outputs)

        return customized_model

    def training_process(self, model, epochs, train_ds, val_ds, test_set):
        model.summary()  # model layers+info

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(),  # most popular optimizer at the moment
            loss="binary_crossentropy",  # loss for classification problem
            metrics=["accuracy"],  # required metric
        )

        # Fit model with data
        # history = model.fit_generator(train_ds, epochs=epochs,
        #                   # callbacks=callbacks,
        #                 validation_data=val_ds,
        #                  )

        lr_finder = LRFinder(model)

        # Train a model with batch size 300 for 5 epochs
        # with learning rate growing exponentially from 0.0001 to 1
        lr_finder.find_generator(train_ds, start_lr=0.000001, end_lr=1, epochs=50)
        # Plot the loss, ignore 20 batches in the beginning and 5 in the end
        lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
        plt.savefig('results/keras/plot_loss.png')
        plt.show()

        # Plot rate of change of the loss
        # Ignore 20 batches in the beginning and 5 in the end
        # Smooth the curve using simple moving average of 20 batches
        # Limit the range for y axis to (-0.02, 0.01)
        lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
        plt.savefig('results/keras/plot_loss_change.png')
        plt.show()

        # Now we are changing learning rate per visualization and recompile the model - fastest decrease on the graph
        learning_rate = float(input('Please type in learning rate per visualization '))
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),  # most popular optimizer at the moment
            loss="binary_crossentropy",  # loss for classification problem
            metrics=["accuracy"],  # required metric
        )
        model.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(
            mode='max',
            patience=10,
            verbose=-1
        )

        # Fit model with data
        history = model.fit_generator(train_ds, epochs=epochs,
                                      callbacks=[early_stopping],
                                      validation_data=val_ds)
        # Model results visualization

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('results/keras/accuracy.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('results/keras/loss.png')
        score = model.evaluate(test_set, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def pretrained_model(self, model, unfreeze):
        if model == 'VGG16':
            model = VGG16(include_top=False, input_shape=self.image_size + (3,), weights='imagenet')

        elif model == 'VGG19':
            model = VGG19(include_top=False, input_shape=self.image_size + (3,), weights='imagenet')

        customized_model = Sequential()

        # Add layers of base model
        for layer in model.layers[:]:
            if unfreeze == 1:
                layer.trainable = True
                customized_model.add(layer)
            elif unfreeze == 0:
                layer.trainable = False
                customized_model.add(layer)

            else:
                raise ValueError('Please choose either 0 - frozen layers or 1 - trainable layers')

        customized_model.add(layers.Flatten())
        customized_model.add(tf.keras.layers.Dropout(0.5))

        # Add last layer with sigmoid activation function since we have 2 classes
        customized_model.add(Dense(units=1, activation='sigmoid'))

        return customized_model


class ImageClassificationPT:
    def __init__(self, path, size):
        self.path = path
        self.size = size

    def load_split_train_test(self, valid_size):
        train_transforms = transforms.Compose([transforms.Resize(self.size),
                                               transforms.ToTensor(),
                                               # Augmentation block
                                               transforms.RandomVerticalFlip(0.4),
                                               transforms.RandomHorizontalFlip(0.4),
                                               transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0,
                                                                      hue=0),
                                               ])
        val_transforms = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             # Augmentation block
                                             transforms.RandomVerticalFlip(0.4),
                                             transforms.RandomHorizontalFlip(0.4),
                                             transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0,
                                                                    hue=0),
                                             ])
        test_transforms = transforms.Compose([transforms.Resize((self.size)),
                                              transforms.ToTensor(),
                                              ])

        train_data = torchvision.datasets.ImageFolder(self.path + '/train',
                                                      transform=train_transforms)
        val_data = torchvision.datasets.ImageFolder(self.path + '/train',
                                                    transform=val_transforms)
        test_data = torchvision.datasets.ImageFolder(self.path + '/test',
                                                     transform=test_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  sampler=train_sampler, batch_size=64)
        valloader = torch.utils.data.DataLoader(val_data,
                                                sampler=val_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

        return trainloader, valloader, testloader

    def visualize_classification(self, trainloader):
        i=1
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            grid = torchvision.utils.make_grid(inputs, nrow=5)
            plt.imshow(transforms.ToPILImage()(grid))
            plt.savefig('results/pytorch/augmented_images_part' + str(i))
            plt.show()
            i+= 1

    def pretrained_model(self, model, unfreeze, trainloader, valloader, testloader, train_epochs):
        device = torch.device("cpu")

        # Architecture part
        if model == 'resnet50':
            basemodel = models.resnet50(pretrained=True)

        elif model == 'densenet':
            basemodel = models.densenet161(pretrained=True)

        else:
            raise ValueError('Model not implemented yet')

        if unfreeze == 1:
            for param in basemodel.parameters():
                param.requires_grad = True

        elif unfreeze == 0:
            for param in basemodel.parameters():
                param.requires_grad = False

        # Resnet
        if model == 'resnet':
            basemodel.fc = nn.Sequential(nn.Linear(2048, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(512, 2),
                                         nn.LogSoftmax(dim=1))
        # Densenet
        else:
            basemodel.classifier = nn.Sequential(nn.Linear(2208, 512),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.1),
                                                 nn.Linear(512, 2),
                                                 nn.LogSoftmax(dim=1))

        optimizer = optim.Adam(basemodel.classifier.parameters(), lr=0.003)
        criterion = nn.NLLLoss()
        print(basemodel.to(device))
        # LR finder
        lr_finder = TLRFinder(basemodel, optimizer, criterion)
        lr_finder.range_test(trainloader, val_loader=valloader, end_lr=1, num_iter=100, step_mode="linear")
        lr_finder.plot(log_lr=False)
        plt.show()
        plt.savefig('results/pytorch/suggested_lr.png')
        lr_finder.reset()
        learning_rate = float(input('Please put the best learning rate you see on the graph '))
        optimizer = optim.Adam(basemodel.classifier.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        #Validation pass function

        # Function for the validation pass
        def validation(model, validateloader, criterion):
            val_loss = 0
            val_accuracy = 0

            for images, labels in iter(valloader):
                images, labels = images.to('cpu'), labels.to('cpu')

                output = model.forward(images)
                val_loss += criterion(output, labels).item()

                probabilities = torch.exp(output)

                equality = (labels.data == probabilities.max(dim=1)[1])
                val_accuracy += equality.type(torch.FloatTensor).mean()
            return val_loss, val_accuracy


        # Train part
        epochs = train_epochs
        steps = 0
        running_loss = 0
        print_every = 10
        train_losses, test_losses = [], []
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = basemodel.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    basemodel.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = basemodel.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            basemodel.eval()
                            val_loss, val_accuracy = validation(basemodel, valloader, criterion)
                    train_losses.append(running_loss / len(trainloader))
                    test_losses.append(test_loss / len(testloader))
                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Val loss: {val_loss / len(valloader):.3f}.. "
                          f"Val accuracy: {val_accuracy / len(valloader):.3f}.. "
                          f"Test loss: {test_loss / len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy / len(testloader):.3f}")
                    running_loss = 0
                    basemodel.train()
        # torch.save(model, 'aerialmodel.pth')#Can uncomment and save the model
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig('results/pytorch/plot_loss_change.png')
        plt.show()
