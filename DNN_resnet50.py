import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
import fidle
from tensorflow.keras.regularizers import l2

h5ftr = h5py.File('features_resnet50_ok.h5', 'r')
features = h5ftr['features'][:]
h5ftr.close()
labels = pd.read_excel('MC_ok.xlsx')
label = labels['MC']
indexes = list(range(len(labels)))
random_states = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
mse_test_list = []
mse_training_list = []
mse_valid_list = []
mae_test_list = []
mae_training_list = []
mae_valid_list = []
Rsquared_test_list = []
Rsquared_training_list = []
for random_state in random_states:
    train_idx, test_idx = train_test_split(indexes, train_size=0.85, shuffle=True, random_state=random_state)
    x_train = features[train_idx]
    x_test = features[test_idx]
    y_train = label[train_idx]
    y_test = label[test_idx]
    def pretrained_f(input, dropOutRate=0.25, hidden_units=4096, num_layers=3):
        x2 = Input(shape=input)
        x2_ = Flatten()(x2)
        for i in range(num_layers):
            x2_ = Dense(hidden_units, activation='elu',
                      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001), bias_regularizer=l2(0.5))(
                x2_)
            x2_ = Dropout(dropOutRate)(x2_)
        x3 = Dense(1, activation="linear")(x2_)
        model = keras.Model(inputs=x2, outputs=x3)
        return model
    fit_verbosity = 1
    # TRAINING MODEL
    # Get it
    model = pretrained_f((2048,))
    model.summary()
    #Add callback
    os.makedirs('.\DEEP_LEARNING', mode=0o750, exist_ok=True)
    save_dir = "best_model.keras"
    savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, save_best_only=True)
    # Train model
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae", "mse"])
    history = model.fit(x_train,
                        y_train,
                        epochs=150,
                        batch_size=16,
                        verbose=fit_verbosity,
                        validation_data=(x_test, y_test),
                        callbacks=[savemodel_callback])
    # Model evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('x_test/loss       : {:5.4f}'.format(score[0]))
    print('x_test/mae       : {:5.4f}'.format(score[1]))
    print('x_test/mse      : {:5.4f}'.format(score[2]))
    print('min(val_mae) : {:5.4f}'.format(min(history.history['val_mae'])))
    fidle.scrawler.history(history, plot={'MSE': ['mse', 'val_mse'],
                                          'MAE': ['mae', 'val_mae'],
                                          'LOSS': ['loss', 'val_loss']}, save_as='01-history')
    mse_tt = float(format(score[2]))
    mae_tt = float(format(score[1]))
    mse_test_list.append(mse_tt)
    mae_test_list.append(mae_tt)
    # Restore model
    loaded_model = tf.keras.models.load_model('best_model.keras')
    loaded_model.summary()
    print("Loaded.")
    #Evaluate model
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('x_test/loss       : {:5.4f}'.format(score[0]))
    print('x_test/mae       : {:5.4f}'.format(score[1]))
    print('x_test/mse      : {:5.4f}'.format(score[2]))
    mse_vd = float(format(score[2]))
    mae_vd = float(format(score[1]))
    mse_valid_list.append(mse_vd)
    mae_valid_list.append(mae_vd)
average_mse_tt = np.mean(mse_test_list)
standard_deviation_mse_tt = np.std(mse_test_list)
average_mae_tt = np.mean(mae_test_list)
standard_deviation_mae_tt = np.std(mae_test_list)
average_mse_vd = np.mean(mse_valid_list)
standard_deviation_mse_vd = np.std(mse_valid_list)
average_mae_vd = np.mean(mae_test_list)
standard_deviation_mae_vd = np.std(mae_valid_list)
print("TEST MSE:", "               Average:", average_mse_tt, "", "      SD :", standard_deviation_mse_tt)
print("TEST MAE:", "               Average:", average_mae_tt, "", "       SD :", standard_deviation_mae_tt)
print("VALIDATION MSE:", "         Average:", average_mse_vd, "", "       SD :", standard_deviation_mse_vd)
print("VALIDATION MAE:", "         Average:", average_mae_vd, "", "       SD :", standard_deviation_mae_vd)