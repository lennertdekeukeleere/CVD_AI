import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler


class CustomWaferGenerator(Sequence):
    # initialize the custom generator
    def __init__(self, df, batch_size, target_height, target_width, X_site, Y_site,conditioning_dim=0):
        self.df = df
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.conditioning_dim = conditioning_dim
        self.X_site = X_site
        self.Y_site = Y_site

    # shuffle the data after each epoch
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)

    # select a batch as tensor
    def __getitem__(self, index):
        cur_df = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(cur_df)
        return X, y

    #
    def __data_generation(self, cur_df):
        # initialize empty tensors to store the images
        X = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 1))
        Y = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 1))
        # initialize emty tensor to store the conditioning variables
        C = np.empty(shape=(self.batch_size, self.conditioning_dim))

        # loop through the current batch and build the tensors
        j = 0
        for i, row in cur_df.iterrows():
            # make interpolation
            layer_thickness = cur_df.loc[i,'SITE_0':]

            grid_x, grid_y = np.mgrid[min_x:max_x:self.target_width*1j, min_y:max_y:self.target_height*1j]
            grid = griddata((self.X_site,self.Y_site), layer_thickness, (grid_x, grid_y), method='cubic')
            idx = np.isnan(grid)
            grid[idx] = 0
            grid = grid.reshape(self.target_width,self.target_height,1)
            X[j] = grid
            Y[j] = grid
            # store conditioning in tensor
            if self.conditioning_dim > 0:
                C[j] = list(cur_df.loc[i,:'DEP TIME'])

            j +=1

        if self.conditioning_dim > 0:
            return [X, C], Y
        else:
            return X, Y

    # get number of batches
    def __len__(self):
        return int(np.floor(self.df.shape[0] / self.batch_size))

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        self.add_loss(kl_loss)
        return reconstruction


def CreateEncoder(latent_dim,conditioning_dim,width,height):
    input_im = layers.Input(shape=(height, width, 1),name="input_im")
    input_cond = layers.Input(shape=(conditioning_dim,),name="input_cond")
    cond_up = layers.Dense(height*width)(input_cond)
    cond_up = layers.Reshape((width, height, 1))(cond_up)
    encoder_inputs = layers.Concatenate(axis=3)([input_im,cond_up])

    x=layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x=layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
    initial_shape = x.shape[1:]
    x=layers.Flatten()(x)
    x = layers.Dense(8, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    z_cond = layers.Concatenate()([z,input_cond])
    encoder = keras.Model(inputs=[input_im,input_cond], outputs=[z_mean, z_log_var, z], name="encoder")
    encoder_cond = keras.Model(inputs=[input_im,input_cond], outputs=[z_mean, z_log_var, z_cond], name="encoder_cond")
    encoder.summary()
    encoder_cond.summary()
    return input_im, input_cond, encoder,encoder_cond, initial_shape

def CreateDecoder(latent_dim,conditioning_dim,decode_shape,width,height):
    latent_inputs = layers.Input(shape=(latent_dim+conditioning_dim,))
    x = layers.Dense(np.prod(decode_shape), activation="relu")(latent_inputs)
    x = layers.Reshape(decode_shape)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
    # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(width*height, activation="sigmoid")(x)
    decoder_outputs = layers.Reshape((width, height, 1))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

def show_sample_image(df,X,Y,width,height,nb=3, verbose=True):
    min_x = X.min()
    max_x = X.max()
    min_y = Y.min()
    max_y = Y.max()
    f, ax = plt.subplots(1, nb, figsize=(10,5))
    for i in range(nb):
        idx = random.choice(list(df.index.values))
        layer_thickness = df.loc[idx,'SITE_0':]
        grid_x, grid_y = np.mgrid[min_x:max_x:width*1j, min_y:max_y:height*1j]
        grid = griddata((X,Y), layer_thickness, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid)
        grid[mask] = 0

        # store conditioning in tensor
        if verbose:
            label = ''
            for col in df.columns:
                if df.loc[idx][col]==1:
                    label = label + '\n' + col
            if nb > 1:
                ax[i].set_xlim(min_x,max_x)
                ax[i].set_ylim(min_y,max_y)
                ax[i].set_xlabel('X [mm]',fontsize=16)
                ax[i].set_ylabel('Y [mm]',fontsize=16)
                ax[i].imshow(grid.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                            origin='lower',aspect='auto',clim=(0,1))
                ax[i].set_title(label)
            else:
                ax.set_xlim(min_x,max_x)
                ax.set_ylim(min_y,max_y)
                ax.set_xlabel('X [mm]',fontsize=16)
                ax.set_ylabel('Y [mm]',fontsize=16)
                ax.imshow(grid.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                            origin='lower',aspect='auto',clim=(0,1))
                ax.set_title(label)

    grid = grid.reshape(width,height,1)
    return grid, list(df.loc[idx,:'DEP TIME'])

def encode_image(img, conditioning, encoder, height, width, batch_size):
    # fill the image to match the batch size
    img_single = np.expand_dims(img, axis=0)
    img_single = img_single.astype(np.float32)
    img_single = np.repeat(img_single, batch_size, axis=0)
    # use the encoder to coomputer the latent space
    if conditioning is None:
        z = encoder.predict(img_single)
    else:
        z = encoder.predict([img_single, np.repeat(np.expand_dims(conditioning, axis=0), batch_size, axis=0)])
    return z

def decode_embedding(z, conditioning, decoder):
    if isinstance(z, list):
        z = np.asarray(z[0])
    conditioning = np.asarray(conditioning)
    z = np.concatenate((z, np.repeat(np.expand_dims(conditioning, axis=0), z.shape[0], axis=0)), axis=1)
    return decoder.predict(z)

def generate_new_images_vae(df,X,Y,latent_dim,cond_dim,width,height,decoder,suffix=''):
    min_x = X.min()
    max_x = X.max()
    min_y = Y.min()
    max_y = Y.max()

    f, ax = plt.subplots(2, 4, figsize=(20,10))
    for i in range(4):
        idx = random.choice(list(df.index.values))
        layer_thickness = df.loc[idx,'SITE_0':]
        grid_x, grid_y = np.mgrid[min_x:max_x:width*1j, min_y:max_y:height*1j]
        grid = griddata((X,Y), layer_thickness, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid)
        grid[mask] = 0
        label = ''
        meta = []
        for col in (df.loc[:,:'DEP TIME']).columns:
            meta.append(df.loc[idx][col])
            label = label + '\n' + col + ' : {:.2f}'.format(df.loc[idx][col])
        label = label + '\n' + 'original image'
        ax[0][i].set_xlim(min_x,max_x)
        ax[0][i].set_ylim(min_y,max_y)
        ax[0][i].set_ylabel('Y [mm]',fontsize=16)
        ax[0][i].imshow(grid.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                    origin='lower',aspect='auto',clim=(0,1))
        ax[0][i].set_title(label)

        z1 = np.random.rand(latent_dim, latent_dim)
        ret = decode_embedding(z1, meta, decoder)
        ret = ret[0].reshape(width,height)
        ax[1][i].set_xlim(min_x,max_x)
        ax[1][i].set_ylim(min_y,max_y)
        ax[1][i].set_xlabel('X [mm]',fontsize=16)
        ax[1][i].set_ylabel('Y [mm]',fontsize=16)
        ax[1][i].imshow(ret.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                    origin='lower',aspect='auto',clim=(0,1))
        ax[1][i].set_title("generated image")

    plt.savefig(figFolder + 'VAE_reconstructions' + suffix + '.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    figFolder = './Figures/'

    if not os.path.exists(figFolder):
        os.makedirs(figFolder)
    # df = pd.read_csv('./Data/test_assignment_sim.csv',\
    # usecols= ['FLOWFACTOR','SPACING','DEP TIME','TOOL','SITE_0'])
    coordinates = pd.read_csv('./Data/site_coordinates.csv')
    # coordinates.plot.scatter('SITE_X','SITE_Y')
    # plt.show()

    X = coordinates['SITE_X']*10**(-3)
    Y = coordinates['SITE_Y']*10**(-3)
    min_x = X.min()
    max_x = X.max()
    min_y = Y.min()
    max_y = Y.max()

    df = pd.read_csv('./Data/test_assignment_sim.csv')
    mask = df['TOOL'] == 1
    df = df[mask]

    mean_flow = df.loc[:,'FLOWFACTOR'].mean()
    std_flow = df.loc[:,'FLOWFACTOR'].std()
    df.loc[:,'FLOWFACTOR'] = (df.loc[:,'FLOWFACTOR']-mean_flow)/(std_flow)
    mean_spacing = df.loc[:,'SPACING'].mean()
    std_spacing = df.loc[:,'SPACING'].std()
    df.loc[:,'SPACING'] = (df.loc[:,'SPACING']-mean_spacing)/(std_spacing)
    mean_deptime = df.loc[:,'DEP TIME'].mean()
    std_deptime = df.loc[:,'DEP TIME'].std()
    df.loc[:,'DEP TIME'] = (df.loc[:,'DEP TIME']-mean_deptime)/(std_deptime)

    min_thickness = df.loc[:,'SITE_0':].min().min()
    max_thickness = df.loc[:,'SITE_0':].max().max()
    df.loc[:,'SITE_0':] = (df.loc[:,'SITE_0':]-min_thickness)/(max_thickness-min_thickness)
    # msk = np.random.rand(len(df)) < 0.8
    # df_train = df[msk]
    # df_val = df[~msk]
    msk = np.sqrt(df['FLOWFACTOR']**2 + df['SPACING']**2 + df['DEP TIME']**2)<2
    df_train = df[msk]
    df_val = df[~msk]

    batch_size = 15
    height = 52
    width = 52
    cond_dim = 3
    latent_dim = 4
    # create generators for training
    gen = CustomWaferGenerator(df_train,
                          batch_size=batch_size,
                          target_height=height,
                          target_width=width,
                          X_site=X,
                          Y_site=Y,
                          conditioning_dim=cond_dim)

    # create generators for validation
    gen_val = CustomWaferGenerator(df_val,
                          batch_size=batch_size,
                          target_height=height,
                          target_width=width,
                          X_site=X,
                          Y_site=Y,
                          conditioning_dim=cond_dim)

    input_im, input_cond, encoder,encoder_cond, initial_shape = CreateEncoder(latent_dim,cond_dim,width,height)
    decoder = CreateDecoder(latent_dim,cond_dim,initial_shape,width,height)

    vae = VAE(encoder_cond, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.MeanSquaredError())
    vae.compile(optimizer="rmsprop",loss=keras.losses.MeanSquaredError())
    # vae.compile(optimizer=keras.optimizers.Adam(),loss="binary_crossentropy")
    # train the variational autoencoder
    vae.fit(gen, verbose=1, epochs=10, validation_data=gen_val)

    grid,meta = show_sample_image(df,X,Y,width,height)

    # display latent space produced by the encoder
    z = encode_image(grid.astype(np.float32),
                     np.array(meta),
                     encoder, height, width, batch_size)
    print('latent sample:\n', z[0][0])

    # reconstruct original image using latent space
    ret = decode_embedding(z, meta, decoder)
    ret = ret[0]
    ret = ret.reshape(width,height)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)
    ax.set_xlabel('X [mm]',fontsize=16)
    ax.set_ylabel('Y [mm]',fontsize=16)
    ax.imshow(ret.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                origin='lower',aspect='auto',clim=(0,1))


    generate_new_images_vae(df_train,X,Y,latent_dim,cond_dim,width,height,decoder,'_train')
    generate_new_images_vae(df_val,X,Y,latent_dim,cond_dim,width,height,decoder,'_val')

    plt.show()
