# lstm_lstm: emotion recognition from speech= lstm, text=lstm
# created for ATSIT paper 2020
# coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)
# changelog:
# 2020/01/28: create names mlp_iemocap_paa
# 2020/08/13: modified for thesis section 412

import numpy as np
import random as rn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd

rn.seed(123)
np.random.seed(99)


def calc_scores(x, y):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x-x_mean)*(y-y_mean))

    x_var = 1.0 / (len(x)-1) * np.nansum((x-x_mean)**2)
    # similar with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y)-1) * np.nansum((y-y_mean)**2)

    CCC = (2*covariance) / (x_var + y_var + (x_mean-y_mean)**2)
    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)
    PCC = covariance / (x_std * y_std)
    RMSE = np.sqrt(np.nanmean((x - y)**2))
    scores = np.array([CCC, PCC, RMSE])
    return scores


# load feature and labels
feat_train = np.load('/home/s1820002/ser_nat/data/feat_68_msp_train.npy')
feat_test  = np.load('/home/s1820002/ser_nat/data/feat_68_msp_test.npy')
feat = np.concatenate((feat_train, feat_test))

list_path = '/home/s1820002/msp-improv/helper/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_file = pd.DataFrame(list_file)
data = list_file.sort_values(by=['wavfile'])

vad_train = []
vad_test = []

for index, row in data.iterrows(): 
    #print(row['wavfile'], row['v'], row['a'], row['d']) 
    if int(row['wavfile'][18]) in range(1,6): 
        #print("Process vad..", row['wavfile']) 
        vad_train.append(row['n']) 
    else:
        #print("Process..", row['wavfile']) 
        vad_test.append(row['n'])

vad_train = np.array(vad_train).reshape(len(vad_train), 1)
vad_test = np.array(vad_test).reshape(len(vad_test), 1)

vad = np.vstack((vad_train, vad_test))

# standardization
scaled_feature = False

if scaled_feature:
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad)
    scaled_vad = scaler.transform(vad)
    vad = scaled_vad
else:
    vad = vad

# LOSO is from utterance 6816
limiter = 6816
X_train = feat[:limiter]
X_test = feat[limiter:]
y_train = vad[:limiter]
y_test = vad[limiter:]

# features reshape
X_train= X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

# batch_size=min(200, n_samples)
# layers (512,256, 128, 64, 32, 16)
nn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16), activation='logistic', 
    solver='adam', alpha=0, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, shuffle=True,
    random_state=9, verbose=1, warm_start=True,
    early_stopping=True, validation_fraction=0.2, 
    n_iter_no_change=10)

nn = nn.fit(X_train, y_train)
y_predict = nn.predict(X_test)

# ccc = []
# pcc = []
# rmse = []

ccc, pcc, rmse = calc_scores(y_predict, y_test)

print("CCC, PCC, RMSE= ", ccc, pcc, rmse)
# print(np.mean(ccc), np.mean(pcc), np.mean(rmse))

# Result
# [0.2603903411341116, 0.5930180580127936, 0.4461756647732577]
