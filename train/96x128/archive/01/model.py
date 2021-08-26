import os, random, time, numpy as np, pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import losses, optimizers, metrics
from tensorflow.keras import Input, Model, layers, callbacks, regularizers
from jarvis.train import custom, datasets, params
from jarvis.train.client import Client
from jarvis.utils.general import gpus, overload, tools as jtools

def valid_generator():
    # --- Create generators for AD/CN 
    client_AD = Client(AD_CLIENT_PATH)
    client_CN = Client(CN_CLIENT_PATH)
    
    gen_train_AD, gen_valid_AD = client_AD.create_generators()
    gen_train_CN, gen_valid_CN = client_CN.create_generators()
    
    while True:
        xs_AD, ys_AD = next(gen_valid_AD)
        xs_CN, ys_CN = next(gen_valid_CN)
        
        # --- Randomize for AD-AD-CN or AD-CN-CN
        choice_index = random.randint(0, 1)
        
        if choice_index == 0:
            xs_final = np.concatenate((xs_AD['dat'], xs_CN['dat'][:1]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'], ys_CN['lbl'][:1]), axis=0)
        else:
            xs_final = np.concatenate((xs_AD['dat'][:1], xs_CN['dat']), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:1], ys_CN['lbl']), axis=0)
            
        yield xs_final, ys_final

def train_generator():
    # --- Create generators for AD/CN 
    client_AD = Client(AD_CLIENT_PATH)
    client_CN = Client(CN_CLIENT_PATH)
    
    gen_train_AD, gen_valid_CN = client_AD.create_generators()
    gen_train_CN, gen_valid_CN = client_CN.create_generators()
    
    while True:
        xs_AD, ys_AD = next(gen_train_AD)
        xs_CN, ys_CN = next(gen_train_CN)
        
        # --- Randomize for AD-AD-CN or AD-CN-CN
        choice_index = random.randint(0, 1)
        
        if choice_index == 0:
            xs_final = np.concatenate((xs_AD['dat'], xs_CN['dat'][:1]), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'], ys_CN['lbl'][:1]), axis=0)
        else:
            xs_final = np.concatenate((xs_AD['dat'][:1], xs_CN['dat']), axis=0)
            ys_final = np.concatenate((ys_AD['lbl'][:1], ys_CN['lbl']), axis=0)
            
        yield xs_final, ys_final

def cosine_similarity(vects):
    """Find the cosine similarity between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing cosine similarity
        (as floating point value) between vectors.
    """
    
    x, y = vects
    
    x = tf.math.l2_normalize(x, axis=1)
    y = tf.math.l2_normalize(y, axis=1)
    return -tf.math.reduce_mean(x * y, axis=1, keepdims=True)

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def prepare_model(inputs, use_cosine_similarity=True, use_normalization=True):
        
    # --- Define lambda functions
    
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_initializer': 'he_uniform'
    }
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    acti = lambda x : layers.LeakyReLU()(x)
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    
    conv1 = lambda filters, x : norm(acti(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : norm(acti(conv(x, filters, strides=(2, 2, 2))))
    tran2 = lambda filters, x : norm(acti(tran(x, filters, strides=(2, 2, 2))))
    
    # --- Define autoencoder network
    
    inp = Input(INPUT_SHAPE)
    e1 = conv1(4, inp)
    e2 = conv1(8, conv2(8, e1))
    e3 = conv1(16, conv2(16, e2))
    e4 = conv1(32, conv2(32, e3))
    e5 = layers.Conv3D(filters=4, kernel_size=(1, 1, 1))(e4)
    e6 = layers.Flatten()(e5)
    e7 = layers.Dense(10, activation='relu', name="ctr")(e6)
    e8 = layers.Dense(1, activation='sigmoid', name="enc")(e7)
    d1 = tran2(16, e4)
    d2 = conv1(8, tran2(8, d1))
    d3 = conv1(4, tran2(8, d2))
    d4 = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), name="dec")(d3)
    
    autoencoder_logits = {}
    autoencoder_logits["ctr"] = e7
    autoencoder_logits["enc"] = e8
    autoencoder_logits["dec"] = d4
    
    autoencoder_network = Model(inputs=inp, outputs=autoencoder_logits)
    
    # --- Define contrastive network
    
    tower_1 = autoencoder_network(inputs[np.newaxis, 0, :])
    tower_2 = autoencoder_network(inputs[np.newaxis, 1, :])
    tower_3 = autoencoder_network(inputs[np.newaxis, 2, :])
    
    if use_cosine_similarity:
        merge_layer1 = layers.Lambda(cosine_similarity)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(cosine_similarity)([tower_2["ctr"], tower_3["ctr"]])
    else:
        merge_layer1 = layers.Lambda(euclidean_distance)([tower_1["ctr"], tower_2["ctr"]])
        merge_layer2 = layers.Lambda(euclidean_distance)([tower_2["ctr"], tower_3["ctr"]])

    if use_normalization:
        normal_layer1 = layers.BatchNormalization()(merge_layer1)
        normal_layer2 = layers.BatchNormalization()(merge_layer2)
    else:
        normal_layer1 = merge_layer1
        normal_layer2 = merge_layer2
    
    siamese_logits = {}
    siamese_logits["ctr1"] = layers.Dense(1, activation="sigmoid", name="ctr1")(normal_layer1)
    siamese_logits["ctr2"] = layers.Dense(1, activation="sigmoid", name="ctr2")(normal_layer2)
    siamese_logits["enc1"] = layers.Layer(name="enc1")(tower_1["enc"])
    siamese_logits["enc2"] = layers.Layer(name="enc2")(tower_2["enc"])
    siamese_logits["enc3"] = layers.Layer(name="enc3")(tower_3["enc"])
    siamese_logits["dec1"] = layers.Layer(name="dec1")(tower_1["dec"])
    siamese_logits["dec2"] = layers.Layer(name="dec2")(tower_2["dec"])
    siamese_logits["dec3"] = layers.Layer(name="dec3")(tower_3["dec"])
    
    siamese = Model(inputs=inputs, outputs=siamese_logits)
    
    return siamese

def train(num_epochs=1, steps_per_epoch=1, debug=False):
    gen_train = train_generator()
    gen_valid = valid_generator()
    
    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch))
        start_time = time.time()

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(gen_train):
            with tf.GradientTape() as tape:
                
                if debug:
                    print("y:", np.squeeze(y_batch_train[0]), np.squeeze(y_batch_train[1]), np.squeeze(y_batch_train[2]))
                    
                logits = model(x_batch_train, training=True)

                # Encoder loss
                enc1_loss = loss_fn['enc1'](y_batch_train[0], logits['enc1'])
                enc2_loss = loss_fn['enc2'](y_batch_train[1], logits['enc2'])
                enc3_loss = loss_fn['enc3'](y_batch_train[2], logits['enc3'])

                # Decoder loss
                dec1_loss = loss_fn['dec1'](x_batch_train[0], logits['dec1'])
                dec2_loss = loss_fn['dec2'](x_batch_train[1], logits['dec2'])
                dec3_loss = loss_fn['dec3'](x_batch_train[2], logits['dec3'])

                # Contrastive loss
                ctr1_loss = loss_fn['ctr1'](y_batch_train[0] == y_batch_train[1], logits['ctr1'])
                ctr2_loss = loss_fn['ctr2'](y_batch_train[1] == y_batch_train[2], logits['ctr2'])
                
                # Combine loss values
                loss_value = enc1_loss + enc2_loss + enc3_loss + dec1_loss + dec2_loss + dec3_loss + ctr1_loss + ctr2_loss

                # Print loss values of current batch
                if debug:
                    print("e1:", K.get_value(enc1_loss), "e2:", K.get_value(enc2_loss), "e3:", K.get_value(enc3_loss))
                    print("d1:", K.get_value(dec1_loss), "d2:", K.get_value(dec2_loss), "d3:", K.get_value(dec3_loss))
                    print("c1:", K.get_value(ctr1_loss), "c2:", K.get_value(ctr2_loss))
                    print("loss:", K.get_value(loss_value))

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
                      
            # Update training metrics
            train_enc1_metric.update_state(y_batch_train[0], logits['enc1'])
            train_enc2_metric.update_state(y_batch_train[1], logits['enc2'])
            train_enc3_metric.update_state(y_batch_train[2], logits['enc3'])        
            
            if debug:
                print("pred:", K.get_value(logits['enc1']), "true:", np.squeeze(y_batch_train[0]))
                print("pred:", K.get_value(logits['enc2']), "true:", np.squeeze(y_batch_train[1]))
                print("pred:", K.get_value(logits['enc3']), "true:", np.squeeze(y_batch_train[2]))

            # Log every steps (batch)
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * p['batch_size']))
                     
            # Quit if step reaches preset steps per epoch
            if step == steps_per_epoch:
                break

        # Display metrics at the end of each epoch
        train_enc1_acc = train_enc1_metric.result()
        train_enc2_acc = train_enc2_metric.result()
        train_enc3_acc = train_enc3_metric.result()
        print("Training acc over epoch:\n", float(train_enc1_acc), "\n", float(train_enc2_acc), "\n", float(train_enc3_acc))

        # Reset training metrics at the end of each epoch
        train_enc1_metric.reset_states()
        train_enc2_metric.reset_states()
        train_enc3_metric.reset_states()

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(gen_valid):
            val_logits = model(x_batch_val, training=False)
            
            # Update val metrics
            val_enc1_metric.update_state(y_batch_val[0], val_logits['enc1'])
            val_enc2_metric.update_state(y_batch_val[1], val_logits['enc2'])
            val_enc3_metric.update_state(y_batch_val[2], val_logits['enc3'])
            
            if step == steps_per_epoch:
                break

        val_enc1_acc = val_enc1_metric.result()
        val_enc2_acc = val_enc2_metric.result()
        val_enc3_acc = val_enc3_metric.result()
        
        val_enc1_metric.reset_states()
        val_enc2_metric.reset_states()
        val_enc3_metric.reset_states()
        
        print("Validation acc:\n", float(val_enc1_acc), "\n", float(val_enc2_acc), "\n", float(val_enc3_acc))
        
        # Output epoch time
        print("Time taken: %.2fs" % (time.time() - start_time))

# =======================================================
# PREPARATION 
# =======================================================

# --- Autoselect GPU
gpus.autoselect()

# --- Define path of clients
AD_CLIENT_PATH = '/data/raw/adni/data/ymls/client-3d-96x128_AD_only.yml'
CN_CLIENT_PATH = '/data/raw/adni/data/ymls/client-3d-96x128_CN_only.yml'
INPUT_SHAPE = (96, 128, 128, 1)

# --- Prepare hyperparams
p = params.load('./hyper.csv', row=0)

MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- Prepare model
inputs = Input(shape=INPUT_SHAPE, name='dat')
model = prepare_model(inputs, use_cosine_similarity=True, use_normalization=False)

# Instantiate an optimizer to train the model
optimizer = optimizers.Adam(learning_rate=p['LR'])

# Instantiate a loss function
loss_fn = {
    'ctr1': loss(),
    'ctr2': loss(),
    'dec1': losses.MeanSquaredError(),
    'dec2': losses.MeanSquaredError(),
    'dec3': losses.MeanSquaredError(),
    'enc1': losses.BinaryCrossentropy(),
    'enc2': losses.BinaryCrossentropy(),
    'enc3': losses.BinaryCrossentropy()
}

# Prepare the metrics
train_enc1_metric = metrics.BinaryAccuracy()
train_enc2_metric = metrics.BinaryAccuracy()
train_enc3_metric = metrics.BinaryAccuracy()
val_enc1_metric = metrics.BinaryAccuracy()
val_enc2_metric = metrics.BinaryAccuracy()
val_enc3_metric = metrics.BinaryAccuracy()

# =======================================================
# TRAINING LOOP 
# =======================================================

steps_per_epoch = 100
num_epochs = p['iterations'] // steps_per_epoch
train(num_epochs=num_epochs, steps_per_epoch=steps_per_epoch, debug=False)
        
# --- Save model
model.save(MODEL_NAME)
