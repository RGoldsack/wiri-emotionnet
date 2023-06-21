
#------------------------------------Importing Packages------------------------------------------>


import pandas as pd
import glob
import numpy as np
from os import system, getcwd, mkdir, remove
from os.path import isdir, getmtime
# import sys
from sys import getsizeof, argv
import time
import random
import json

# from matplotlib import pyplot
from math import sqrt, floor, pi, log2, exp, ceil
from statistics import mean

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)
tf.compat.v1.enable_eager_execution()

from tensorflow.python.ops import math_ops
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from Import import importFiles, model_import


#------------------------------------Functions------------------------------------------>


def reshapeArray(array, batch_size = None, dim = 3):
    
#    array = array[:(floor(array.shape[0] / batch_size) * batch_size)]

    array = np.array(array)
    if dim == 3:
        array = array.reshape(int(array.shape[0]/1), 1, array.shape[1])
        return array
    elif dim == 2:
        array = array.reshape(array.shape[0], array.shape[1])
        return array
    
class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
#    def on_epoch_begin(self, epoch, logs=None):
#        tf.print(os.system("nvidia-smi"))
        
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        tf.print('\nGPU memory details [current: {} GB, peak: {} GB]'.format(
        round(float(gpu_dict['current']) / (1024 ** 3), 4),
        round(float(gpu_dict['peak']) / (1024 ** 3), 4)))
    
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def get_dataset(df, emoMeasure = "all", physMeasure = "both"):
    if (emoMeasure == "") or (physMeasure == ""):
        raise ValueError("Measure type must not be empty.")
    
    if emoMeasure == "6emo":
        y = df.filter(["Anger.Self", "Disgust.Self", "Fear.Self", "Happiness.Self", "Sadness.Self", "Surprise.Self"], axis = 1)
        y_cols = y.columns
    if emoMeasure == "indiv.PANAS":
        y = df.filter(regex = "PANAS\.", axis = 1)
        y = y.drop(["PANAS.Positive", "PANAS.Negative"], axis = 1)
        y_cols = y.columns
    if emoMeasure == "sum.PANAS":
        y = df.filter(["PANAS.Positive", "PANAS.Negative"], axis = 1)
        y_cols = y.columns
    if emoMeasure == "cont":
        y = df.filter(["Response.Self"], axis = 1)
        y_cols = y.columns
    if emoMeasure == "random.walk":
        step_n = 81920
        path = np.zeros((step_n))
        for i in range(1, step_n):
            path[i] = path[i - 1] + np.random.normal(loc = 0, scale = 0.25, size = 1)
            if path[i] > 10:
                path[i] = 10
            if path[i] < 1:
                path[i] = 1

        y = path.reshape(step_n,1)
        y_cols = ["Random Walk"]
    if physMeasure == "mocap":
        X = df.filter(regex = "\.Bone\.", axis = 1)
        X_cols = X.columns
    if physMeasure == "phys":
        X = df.filter(["HR", "ECG", "ChestEx", "SkinTemp", "GSR"], axis = 1)
        X_cols = X.columns
    if physMeasure == "both":
        X = df.filter(regex = "\.Bone\.|HR|ECG|ChestEx|SkinTemp|GSR", axis = 1)
        X_cols = X.columns
    if physMeasure == "random.N":
        step_n = 81920
        cols = 50
        X = np.zeros((step_n, cols))
        X_cols = []
        
        for i in range(1, step_n):
            X[i, :] = X[i - 1, :] + np.random.normal(loc = 0, scale = 0.25, size = cols)
        for col in range(X.shape[1]):
            X[:, col] = X[:, col] * (col * 0.1)
            X_cols.append("Col" + str(col+1))
            
    X, y = np.array(X), np.array(y)
    
    print(y.shape)

    return X, y, X_cols, y_cols


def layer_y(x, outputDenseSize_L1 = 512, outputDenseSize_L2 = 256, dropout = 0.2, size = None, activation = "softmax", name = "NO NAME"):
    y = Dense(outputDenseSize_L1, activation = LeakyReLU())(x)
    y = Dropout(dropout)(y)
    y = Dense(outputDenseSize_L2, activation = LeakyReLU())(y)
    y = Dropout(dropout)(y)
    
    y = Dense(size, activation = activation, name = name)(y)
    
    return y

def wrapper(path = None, cur_location = None):
    # print("CURRENT LOCATION", cur_location)
    def contLoss(y_true, y_pred):
        # Getting current epoch
        if getcwd() == "C:\\Users\\golds\\Downloads":
            path = "C:/Users/golds/OneDrive/Desktop/DyadFiles/" + "Results/"
        if getcwd() == "/nfs/home/goldsaro":
            path = "/nfs/scratch/goldsaro/DyadFiles/" + "Results/"
        epoch = max(glob.glob(path + "Epochs/" + cur_location  + "epoch*"), key = getmtime)
        epoch = int(epoch[len(path + "Epochs/" + cur_location + "epoch"):].split(".")[0])
        
        y_true = float(y_true)
        pathAll = path + "Temp/" + cur_location
        num = 0
        
        # Getting batch # within epoch
        if epoch > 1:
            file_num = max(glob.glob(pathAll + "temp*"), key = getmtime)
            file_num = file_num[len(pathAll):]
            file_num = file_num.split("_")[1].split(".")[0]
            num = int(file_num) + 1
        # Before we have a stable sigma we use the current estimate and export the current y_true and y_pred
        if epoch < 6:
            sigma = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = 0))
            out_path = pathAll + "temp" + str(epoch) + "_" + str(num) + ".csv"
            out = np.append(y_true, y_pred, axis = 1)
            out = pd.DataFrame(out, columns = ["y_true", "y_pred"])
            out.to_csv(out_path, index = False)
        else: # Creating the stable value of sigma for all epochs after epoch 5 by reading in previous y_true and y_pred - but only if the df of the previous y_true and y_pred has not been created yet (i.e. only at epoch 6)
            if len(glob.glob(path + "Sigma/" + cur_location + "sigma*")) == 0:
                li = []
                for filename in glob.glob(pathAll + "temp*"):
                    df = pd.read_csv(filename, index_col = None, header=0)
                    li.append(df)
                df = pd.concat(li, axis = 0, ignore_index = True)
                sigma = tf.math.sqrt(tf.math.reduce_mean(tf.square(df["y_true"] - df["y_pred"]), axis = 0))
                sigma = sigma.numpy()
                
                filename = path + "Sigma/" + cur_location + "sigma.txt"
                np.savetxt(filename, [sigma])
                sigma = np.loadtxt(filename)
            else: # If the df of previous y_true and y_pred has been created, import this and calculate stable sigma from it
                sigma = np.loadtxt(path + "Sigma/" + cur_location + "sigma.txt")
                
        loss = tf.math.reduce_mean((1/2) * log2(2 * pi) + log2(sigma) + (1/2) * (tf.square((y_true - y_pred)/sigma)), axis = 0)
        # print(loss)
        
        return loss
    return contLoss

def multi_model(n_inputs, n_outputs, length, hidden_nodes, runtimeOptions, dataOptions, cur_location):
    
    # Specify Hyperparameter Options
    outputDenseSize_L1 = 512
    outputDenseSize_L2 = 256
    # outputDenseActivation_L1 = "relu" # try relu or leaky relu
    outputDenseActivation_L2 = "softmax" # change to softmax for all but continuous rating
    dropout = 0.2
    
    # define model
    inputs = Input(shape = (None, n_inputs), batch_size = runtimeOptions[1])

    # let's add a fully-connected layer
    x = Sequential()(inputs)
    x = LSTM(hidden_nodes,
             stateful = runtimeOptions[3],
             return_sequences = False,
             activation = "tanh")(x)
    x = Dense(n_inputs, activation = LeakyReLU())(x)
    x = Dropout(dropout)(x)
    x = Dense(n_inputs, activation = LeakyReLU())(x)
    x = Dropout(dropout)(x)

    # start passing that fully connected block output to all the
    # different model heads
    if dataOptions[1] == "cont":          ###################### Continuous Rating ##############
        y1 = layer_y(x, size = 1, activation = "relu", name = "ContinuousRating") # try changing activation to linear
        
        model = Model(
            inputs = inputs,
            outputs = [y1],
            name = "emotionnet"
        )

        # compile model
        model.compile(optimizer = Adam(amsgrad = True),
                      loss = wrapper(cur_location = cur_location))

        model.summary()

        return model
    if dataOptions[1] == "random.walk":         ###################### Random Walk ##############
        y1 = layer_y(x, size = 1, activation = "relu", name = "RandomWalk")
        
        model = Model(
            inputs = inputs,
            outputs = [y1],
            name = "emotionnet"
        )

        # compile model
        
        model.compile(optimizer = Adam(amsgrad = True),
                      loss = wrapper(cur_location = cur_location))
        model.summary()

        return model
    elif dataOptions[1] == "sum.PANAS":   ############################## Sum PANAS ##############
        size = int(40)
        y1 = layer_y(x, size = size, name = "Positive")
        y2 = layer_y(x, size = size, name = "Negative")
        
        model = Model(
            inputs = inputs,
            outputs = [y1, y2],
            name = "emotionnet"
        )
    elif dataOptions[1] == "6emo":        ############################# 6 Emotions ##############
        
        size = int(5)
        y1 = layer_y(x, size = size, name = "Anger")
        y2 = layer_y(x, size = size, name = "Disgust")
        y3 = layer_y(x, size = size, name = "Fear")
        y4 = layer_y(x, size = size, name = "Happiness")
        y5 = layer_y(x, size = size, name = "Sadness")
        y6 = layer_y(x, size = size, name = "Surprise")
        
        model = Model(
            inputs = inputs,
            outputs = [y1, y2, y3, y4, y5, y6],
            name = "emotionnet"
        )
        
    elif dataOptions[1] == "indiv.PANAS": ####################### Individual PANAS ##############
        
        size = int(5)
        y1  = layer_y(x, size = size, name = "Interested")
        y2  = layer_y(x, size = size, name = "Distressed")
        y3  = layer_y(x, size = size, name = "Excited")
        y4  = layer_y(x, size = size, name = "Upset")
        y5  = layer_y(x, size = size, name = "Strong")
        y6  = layer_y(x, size = size, name = "Guilty")
        y7  = layer_y(x, size = size, name = "Scared")
        y8  = layer_y(x, size = size, name = "Hostile")
        y9  = layer_y(x, size = size, name = "Enthusiastic")
        y10 = layer_y(x, size = size, name = "Proud")
        y11 = layer_y(x, size = size, name = "Irritable")
        y12 = layer_y(x, size = size, name = "Alert")
        y13 = layer_y(x, size = size, name = "Ashamed")
        y14 = layer_y(x, size = size, name = "Inspired")
        y15 = layer_y(x, size = size, name = "Nervous")
        y16 = layer_y(x, size = size, name = "Determined")
        y17 = layer_y(x, size = size, name = "Attentive")
        y18 = layer_y(x, size = size, name = "Jittery")
        y19 = layer_y(x, size = size, name = "Active")
        y20 = layer_y(x, size = size, name = "Afraid")
        
        model = Model(
            inputs = inputs,
            outputs = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20],
            name = "emotionnet"
        )
        
    # define losses

    # compile model
    # https://datascience.stackexchange.com/questions/10523/guidelines-for-selecting-an-optimizer-for-training-neural-networks
    model.compile(optimizer = Adam(amsgrad = True),
                  loss = "CategoricalCrossentropy",
                  metrics = ["CategoricalAccuracy"])

    model.summary()
    
    return model

def evaluate_model(X, y, y_cols, dataOptions, runtimeOptions):
    start_time = time.time()
    results = []
    
    # define evaluation procedure
    cv = RepeatedKFold(n_splits = 10, n_repeats = 1, random_state = 1)
    # enumerate folds
    cvNum = 1
    for train_ix, test_ix in cv.split(X):
        start_timeCV = time.time()
        print("\n")
        print("-------------", "Cross-Validation Fold", cvNum, "of 10", "-------------", "\n")
        # loss prep
        cur_location = str(dataOptions[0]) + "_" + str(dataOptions[1]) + "_" + str(dataOptions[2]) + "_" + str(runtimeOptions[4]) + "_V" + str(runtimeOptions[5]) + "_F" + str(params[6]) + "/"
        # print(cur_location)
        
        dir_list = [path + "Results/Epochs/" + cur_location,
                    path + "Results/Temp/"   + cur_location,
                    path + "Results/Sigma/"  + cur_location]
        for direc in dir_list:
            if isdir(direc):
                for f in glob.glob(direc + "*"):
                    remove(f)
            else: mkdir(direc)
        
        # prepare training and test data for CV step
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # Split the remaining data to train and validation
        pc_val = 0.1
        X_train, X_val = train_test_split(X_train,
                                          test_size = (pc_val/0.9),
                                          shuffle = False,
                                          random_state = 123)
        y_train, y_val = train_test_split(y_train,
                                          test_size = (pc_val/0.9),
                                          shuffle = False,
                                          random_state = 123)
        
        print("X_train shape before reshape:   ", X_train.shape)
        print("y_train shape before reshape:   ", y_train.shape)
        print("X_test shape before reshape:    ", X_test.shape)
        print("y_test shape before reshape:    ", y_test.shape)
        print("X_val shape before reshape:     ", X_val.shape)
        print("y_val shape before reshape:     ", y_val.shape)
                
        X_train = reshapeArray(X_train, runtimeOptions[1])
        X_test  = reshapeArray(X_test,  runtimeOptions[1])
        X_val   = reshapeArray(X_val,   runtimeOptions[1])
        
        n_inputs, n_outputs, length = X_train.shape[2], y_train.shape[1], X_train.shape[0]
        
        print("X_train shape after reshape:    ", X_train.shape)
        print("y_train shape after reshape:    ", y_train.shape)
        print("X_test shape after reshape:     ", X_test.shape)
        print("y_test shape after reshape:     ", y_test.shape)
        print("X_val shape after reshape:      ", X_val.shape)
        print("y_val shape after reshape:      ", y_val.shape)
        
        if runtimeOptions[4] == "rand":
            ls_rand = {"cont": [1, 10], "6emo": [1, 5], "sum.PANAS": [10, 50], "indiv.PANAS": [1, 5], "random.walk": [1, 10]}
            y_train = np.random.randint(ls_rand[dataOptions[1]][0], ls_rand[dataOptions[1]][1], [X_train.shape[0], int(len(y_cols))])
            y_test  = np.random.randint(ls_rand[dataOptions[1]][0], ls_rand[dataOptions[1]][1], [X_test.shape[0],  int(len(y_cols))])
            y_val   = np.random.randint(ls_rand[dataOptions[1]][0], ls_rand[dataOptions[1]][1], [X_val.shape[0],   int(len(y_cols))])
            
        
        # calculate the number of hidden nodes in LSTM
        hidden_nodes = round((2/3) * (n_inputs + n_outputs))

        # printing model information
        print("\n")
        print("---------- Current Model Information ----------")
        print("Part", (dataOptions[0] + 1), "of 8")
        print("Emotion Measure:           ", dataOptions[1])
        print("Bodily Activity Measure:   ", dataOptions[2])
        print("Hidden Nodes:              ", hidden_nodes)
        print("Inputs:                    ", n_inputs)
        print("Outputs:                   ", n_outputs)
        print("Length:                    ", length, "Rows /", runtimeOptions[1], "Batch Size")
        print("Epochs:                    ", runtimeOptions[0])
        print("Output Type:               ", runtimeOptions[4])
        print("Valence:                   ", runtimeOptions[5])
        print("Frequency:                 ", runtimeOptions[6])
        
        # seperate y-values into a list of arrays to be fed to dense layers seperately
        ls = {"6emo": 5, "sum.PANAS": 40, "indiv.PANAS": 5}
        y_train_seperate = []
        y_val_seperate   = []
        y_test_seperate  = []

        for i in range(n_outputs):
            y_train_seperate.append(y_train[:, i])
            y_val_seperate.append(  y_val[:,   i])
            y_test_seperate.append( y_test[:,  i])
            if dataOptions[1] != "cont":
                if dataOptions[1] != "random.walk":
                    y_train_seperate[i] = tf.one_hot((y_train_seperate[i] - 1), depth = ls[dataOptions[1]])
                    y_val_seperate[i]   = tf.one_hot((y_val_seperate[i]   - 1), depth = ls[dataOptions[1]])
                    y_test_seperate[i]  = tf.one_hot((y_test_seperate[i]  - 1), depth = ls[dataOptions[1]])
                
        # create model
        model = multi_model(n_inputs, n_outputs, length, hidden_nodes, runtimeOptions, dataOptions, cur_location = cur_location)
        model.run_eagerly = True
        
        # fit model
        historydf = pd.DataFrame()
        # es = callbacks.EarlyStopping(monitor = "val_loss", min_delta = 5, patience = 5, verbose = 1)
        time_callback = TimeHistory()
        for i in range(runtimeOptions[0]):
            tf.print("For Epoch", i + 1, "of", runtimeOptions[0], "in CV Fold", cvNum)
            
            filename = path + "Results/Epochs/" + cur_location + "epoch" + str(i + 1)
            myText = open("%s.txt" % filename, "wt")
            myText.write(str(i+1))
            myText.close()
            
            history = model.fit(x = X_train,
                                y = y_train_seperate,
                                batch_size = runtimeOptions[1],
                                verbose = runtimeOptions[2],
                                epochs = 1,
                                validation_data = (X_val, y_val_seperate),
                                callbacks = [time_callback, MemoryPrintingCallback()],
                                shuffle = False)
            
            
            
            historydf2 = pd.DataFrame(history.history)
            historydf = pd.concat([historydf, historydf2])
            model.reset_states()
        times = time_callback.times
#        pyplot.plot(np.array(range(runtimeOptions[0])), historydf['loss'], label='train')
#        pyplot.plot(np.array(range(runtimeOptions[0])), historydf['val_loss'], label='val')
#        pyplot.legend()
#        pyplot.show()
#        plot_name = path + "Results/" + "plot" + "_" + str(cvNum) + "_" + str(dataOptions[0]) + "_" + str(dataOptions[1]) + "_" + str(dataOptions[2]) + "_" + str(runtimeOptions[4]) + "_V" + str(runtimeOptions[5]) + ".png"
#        pyplot.savefig(plot_name)

        
        # evaluate the model on test set
        error = model.evaluate(X_test,
                               y_test_seperate,
                               verbose = runtimeOptions[2],
                               batch_size = runtimeOptions[1])
        print("Test Results:", error)
        
        # getting actual vs. predicted
        yhat = model.predict(X_test, batch_size = runtimeOptions[1])
        if (dataOptions[1] == "cont") or (dataOptions[1] == "random.walk"):
            yhat = np.array(yhat)
        else:
            yhat_list = []
            for i in range(n_outputs):
                yhat_list.append(np.argmax(yhat[i], axis = 1))
            yhat = np.array(yhat_list) + 1
            yhat = np.transpose(yhat)
        y_PrAc = np.append(yhat, y_test, axis = 1)
        print("y_PrAc shape", y_PrAc.shape)
        y_cols2, PrAc = list(y_cols) * 2, (["Predicted"] * int(len(y_cols))) + (["Actual"] * int(len(y_cols)))
        for col in range(len(y_cols2)): y_cols2[col] = y_cols2[col] + "_" + PrAc[col]
        y_PrAc = pd.DataFrame(y_PrAc, columns = y_cols2)

        path_y_PrAc = path + "Results/" + "y_PrAc" + "_" + str(cvNum) + "_" + str(dataOptions[0]) + "_" + str(dataOptions[1]) + "_" + str(dataOptions[2]) + "_" + str(runtimeOptions[4]) + "_V" + str(runtimeOptions[5]) + "_F" + str(params[6]) + ".csv"
        y_PrAc.to_csv(path_y_PrAc)
        
        # store result
        results.append(error)
        
        end_timeCV = time.time()
        hours, rem = divmod(end_timeCV - start_timeCV, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Cross validation split", str(cvNum), "done. It took", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        cvNum += 1
        
    print(results)
    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    headers = ["OverallError"]
    if (dataOptions[1] == "cont") or (dataOptions[1] == "random.walk"):
        results = pd.DataFrame(results, columns = headers)
    else:
        y_cols2, cols = list(y_cols) * 2, (["Error"] * int(len(y_cols))) + (["Accuracy"] * int(len(y_cols)))
        for col in range(len(y_cols2)): y_cols2[col] = y_cols2[col] + "_" + cols[col]
        headers = headers + y_cols2
        results = pd.DataFrame(results, columns = headers)
    
    return results

def LSTM_run(dataset, dataOptions, runtimeOptions):
    print("----------- Importing Data ------------")
    print("Dataset Shape", dataset.shape)
    print(round(getsizeof(dataset)/1e+6, 2), "MB")

    print("----------- Getting Dataset -----------")
    X, y, X_cols, y_cols = get_dataset(dataset, dataOptions[1], dataOptions[2])
    print("X size:", round(getsizeof(X)/1e+6, 2), "MB")
    print("y size:", round(getsizeof(y)/1e+6, 2), "MB")
    if runtimeOptions[4] == "shuf":
        y = shuffle(y, random_state = 123)
        
    print("Part", (dataOptions[0] + 1), "of 8")
    print("Using the emotion measure:  ", dataOptions[1])
    print("Using the phys measure:     ", dataOptions[2])
    
    
    print("----------- Running Model -------------")
    results = evaluate_model(X, y, y_cols, dataOptions, runtimeOptions)
    return results

def LSTM_big(path, dataOptions, runtimeOptions, sectionList = None, random_order = False):
    print("----------- LSTM Big ------------------")
    
    
    dyadList = sectionList[dataOptions[0]]
    dataset = model_import(path, dyadList, valence = runtimeOptions[5])
    dataset = dataset.dropna()
    
    if dataOptions[1] == "cont":
        if runtimeOptions[6] != "33.33L":
            dataset["Time"] = pd.DataFrame({"Time": range(dataset.shape[0])}) * 1/30
            dataset["Time"] = pd.to_datetime(  dataset["Time"], unit = "s")
            dataset["Time"] = pd.DatetimeIndex(dataset["Time"], dtype = "datetime64[ns]")
            dataset = dataset.set_index("Time", drop = False)
            dataset = dataset.resample(runtimeOptions[6]).mean()
            dataset = dataset.reset_index(drop = True)
            dataset.index = dataset.index.astype(int)
    
    # rounding # of rows to the previous 10
    dataset = dataset.head(floor(dataset.shape[0] / (runtimeOptions[1] * 10)) * (runtimeOptions[1] * 10))
    dataset = dataset.reset_index(drop = True)
    
    results = LSTM_run(dataset, dataOptions, runtimeOptions)
    
    results = pd.DataFrame(results)
    pathRes = path + "Results/" + "Results" + "_" + str(dataOptions[0]) + "_" + str(dataOptions[1]) + "_" + str(dataOptions[2]) + "_" + str(runtimeOptions[4]) + "_V" + str(runtimeOptions[5]) + "_F" + str(params[6]) + ".csv"
#   pathRes = path + "Results/" + "Results" + "_" + "section"           + "_" + "emotion"           + "_" + "phys"              + "_" + "random"               + "_V" + "valence"              + "_F" + "frequency"    + ".csv"
    results.to_csv(pathRes)

    
# path = "/Volumes/fastt/Data/DyadFiles/"

# section = glob.glob(path + "D*")
# for dyad in range(len(section)):
#     section[dyad] = section[dyad][-13:]
# seed(5)
# shuffle(section)
# partition(section, 8)

with open("params.txt", "r") as fp:
    params = json.load(fp)
params = params[argv[1]]

print("Job: ", params[0])
print(getcwd())



if getcwd() == "C:\\Users\\golds\\Downloads":
    print("--------- Running Locally on Roydons Machine ---------\n\n")
    path = "C:/Users/golds/OneDrive/Desktop/DyadFiles/"
    section0 = ['D31_dfBig.csv']
    n_epochs = {"phys": 250, "mocap": 100, "both": 100, "random.N": 500}
    
    dataOptions = [int(0),                 # section         [0]
                   str("cont"),            # emotion         [1] - "6emo", "cont", "indiv.PANAS", "sum.PANAS", "random.walk"
                   str("phys")]            # phys            [2] - "both", "phys", "mocap", "random.N"
    
    runtimeOptions = [n_epochs[dataOptions[2]],                 # epochs          [0]
                      2048,                # batch size      [1]
                      2,                   # verbose         [2] - 0, 1, 2
                      True,                # stateful        [3] - True, False
                      "observed",              # random          [4] - "shuf", "rand", "observed"
                      "both"]              # valence         [5] - "both", "pos", "neg"
                      
    section0 = ['D34_dfBig.csv', 'D35_dfBig.csv', 'D09_dfBig.csv', 'D30_dfBig.csv', 'D53_dfBig.csv']
    section1 = ['D31_dfBig.csv', 'D47_dfBig.csv', 'D43_dfBig.csv', 'D32_dfBig.csv', 'D20_dfBig.csv']
    section2 = ['D25_dfBig.csv', 'D48_dfBig.csv', 'D52_dfBig.csv', 'D40_dfBig.csv', 'D06_dfBig.csv']
    section3 = ['D36_dfBig.csv', 'D13_dfBig.csv', 'D19_dfBig.csv', 'D24_dfBig.csv', 'D16_dfBig.csv']
    section4 = ['D03_dfBig.csv', 'D38_dfBig.csv', 'D29_dfBig.csv', 'D42_dfBig.csv']
    section5 = ['D28_dfBig.csv', 'D23_dfBig.csv', 'D18_dfBig.csv', 'D39_dfBig.csv', 'D51_dfBig.csv']
    section6 = ['D22_dfBig.csv', 'D44_dfBig.csv', 'D21_dfBig.csv', 'D08_dfBig.csv', 'D26_dfBig.csv']
    section7 = ['D41_dfBig.csv', 'D04_dfBig.csv', 'D45_dfBig.csv', 'D33_dfBig.csv', 'D27_dfBig.csv']

    sectionList = [section0, section1, section2, section3, section4, section5, section6, section7]
    
elif (getcwd() == "/nfs/home/goldsaro") or (getcwd() == "/scale_wlg_persistent/filesets/home/goldsaro"):
    if getcwd() == "/nfs/home/goldsaro":
        print("--------- Running in the Cloud on Raapoi ---------\n\n")
        path = "/nfs/scratch/goldsaro/DyadFiles/"
        
        section0 = ['D34_dfBig.csv', 'D35_dfBig.csv', 'D09_dfBig.csv', 'D30_dfBig.csv', 'D53_dfBig.csv']
        section1 = ['D31_dfBig.csv', 'D47_dfBig.csv', 'D43_dfBig.csv', 'D32_dfBig.csv', 'D20_dfBig.csv']
        section2 = ['D25_dfBig.csv', 'D48_dfBig.csv', 'D52_dfBig.csv', 'D40_dfBig.csv', 'D06_dfBig.csv']
        section3 = ['D36_dfBig.csv', 'D13_dfBig.csv', 'D19_dfBig.csv', 'D24_dfBig.csv', 'D16_dfBig.csv']
        section4 = ['D03_dfBig.csv', 'D38_dfBig.csv', 'D29_dfBig.csv', 'D42_dfBig.csv']
        section5 = ['D28_dfBig.csv', 'D23_dfBig.csv', 'D18_dfBig.csv', 'D39_dfBig.csv', 'D51_dfBig.csv']
        section6 = ['D22_dfBig.csv', 'D44_dfBig.csv', 'D21_dfBig.csv', 'D08_dfBig.csv', 'D26_dfBig.csv']
        section7 = ['D41_dfBig.csv', 'D04_dfBig.csv', 'D45_dfBig.csv', 'D33_dfBig.csv', 'D27_dfBig.csv']

        sectionList = [section0, section1, section2, section3, section4, section5, section6, section7]
        
        
    elif getcwd() == "/scale_wlg_persistent/filesets/home/goldsaro":
        print("--------- Running in the Cloud on NeSI   ---------\n\n")
        path = "/nesi/project/vuw03705/DyadFiles/"
    
        section0 = ['D34_dfBig.csv', 'D35_dfBig.csv', 'D09_dfBig.csv', 'D30_dfBig.csv', 'D53_dfBig.csv']
        section1 = ['D31_dfBig.csv', 'D47_dfBig.csv', 'D43_dfBig.csv', 'D32_dfBig.csv', 'D20_dfBig.csv']
        section2 = ['D25_dfBig.csv', 'D48_dfBig.csv', 'D52_dfBig.csv', 'D40_dfBig.csv', 'D06_dfBig.csv']
        section3 = ['D36_dfBig.csv', 'D13_dfBig.csv', 'D19_dfBig.csv', 'D24_dfBig.csv', 'D16_dfBig.csv']

        sectionList = [section4, section5, section6, section7]
    
    n_epochs   = {"phys": 100, "mocap": 100, "both": 100}
    
    batch_dict = {"33.33L": 2048, "66.66L": 1024, "1S": 512, "5S": 256, "10S": 128}
    batch_size = batch_dict[str(params[6])] if str(params[2]) == "cont" else 2048
    
    dataOptions = [int(params[1]),                   # section         [0] - 1, ..., 10
                   str(params[2]),                   # emotion         [1] - "6emo", "cont", "indiv.PANAS", "sum.PANAS"
                   str(params[3])]                   # phys            [2] - "both", "phys", "mocap"
    
    runtimeOptions = [n_epochs[dataOptions[2]],      # epochs          [0]
                      batch_size,                    # batch size      [1]
                      2,                             # verbose         [2] - 0, 1, 2
                      True,                          # stateful        [3] - True, False
                      str(params[4]),                # random          [4] - "shuf", "rand", "observed"
                      str(params[5]),                # valence         [5] - "both", "pos", "neg"
                      str(params[6])]                # cont frequency  [6] - "33.33L", "66.66L", "1S", "5S", "10S"






LSTM_big(path = path, sectionList = sectionList, dataOptions = dataOptions, runtimeOptions = runtimeOptions)
