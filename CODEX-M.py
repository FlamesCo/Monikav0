
import os
import sys
import time

from PIL import Image

import numpy as np

from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Embedding, concatenate, Conv2DTranspose, ZeroPadding2D, UpSampling2D, Conv2D, Activation, BatchNormalization
from keras.optimizers import Adam

 
# def wasserstein_loss(y_true, y_pred):
#     return K.mean(y_true * y_pred)
## add varaibles to touch the classes on allof the images
latent_input = Input(shape=(100,))

    x = Dense(128 * 16 * 16, activation='relu')(latent_input)
    x = Reshape((16, 16, 128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(3, kernel_size=3, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(latent_input, x)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy']):  # 0.0002, 0.5
# finish the program and let it draw the anime character of your choice
    # seed = int(sys.argv[1])
    # nb_epoch = int(sys.argv[2])
    # nb_batch = int(sys.argv[3])
    # nb_sample = int(sys.argv[4])
    # nb_class = int(sys.argv[5])
    seed = 1
    nb_epoch = 1
    nb_batch = 4
    nb_sample = 1
    nb_class = 1

    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # tf.set_random_seed(seed)

    # load the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # build the generator
    build_generator=True
    generator = build_generator()
    # build the discriminator
    build_discriminator=True
    discriminator = build_discriminator()

    # build the GANs
    gan_input = Input(shape=(100,))
    gan_model= generator(gan_input)
    gan = gan_model(discriminator, generator)

    # train
    gan.fit(X_train, X_train, batch_size=nb_batch, epochs=nb_epoch)

    # save the generator model
    generator.save('generator.h5')

    # save the discriminator model
    discriminator.save('discriminator.h5')() # 0.5, 0.5