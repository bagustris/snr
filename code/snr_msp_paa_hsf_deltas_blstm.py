# lstm_lstm: emotion recognition from speech= lstm, text=lstm
# created for ATSIT paper 2020
# coded by Bagus Tris Atmaja (bagus@ep.its.ac.id)
# changelog:
# 2020/01/28: create names mlp_iemocap_paa
# 2020/08/13: modified for thesis section 412
# 2020/12/15: modified for SER_NAT project


import numpy as np
import random as rn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, CuDNNLSTM, BatchNormalization, Dense, Input
from keras.layers import Dropout, Bidirectional
from keras.models import Model
import keras.backend as K

rn.seed(123)
np.random.seed(99)

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss


# load feature and labels
feat_train = np.load('/home/s1820002/disertasi-ser/data/feat_paa_msp_train_deltas.npy')
feat_test = np.load('/home/s1820002/disertasi-ser/data/feat_paa_msp_test_deltas.npy')
feat = np.concatenate((feat_train, feat_test))

list_path = '/home/s1820002/msp-improv/helper/improv_data.csv'
list_file = pd.read_csv(list_path, index_col=None)
list_file = pd.DataFrame(list_file)
data = list_file.sort_values(by=['wavfile'])

vad_train = []
vad_test = []

for index, row in data.iterrows(): 
    #print(row['wavfile'], row['v'], row['a'], row['d']) 
    if int(row['wavfile'][18]) in range(1, 6): 
        #print("Process vad..", row['wavfile']) 
        vad_train.append(row['n']) 
    else:
        #print("Process..", row['wavfile']) 
        vad_test.append(row['n'])

vad = np.hstack((vad_train, vad_test))

# standardization
scaled_feature = True

if scaled_feature:
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

feat = feat.reshape(feat.shape[0], 1, feat.shape[1])

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaler = scaler.fit(vad.reshape(-1, 1))
    # .reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad.reshape(-1, 1))
    vad = scaled_vad
else:
    vad = vad

# LOSO is from utterance limiter
limiter = 6816

## API model, if use RNN, first two rnn layer mus, activation='relu' <-- No
def api_model():
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    # net_speech = CuDNNLSTM(512, return_sequences=True)(net_speech)
    # net_speech = CuDNNLSTM(256, return_sequences=True)(net_speech)
    # net_speech = CuDNNLSTM(128, return_sequences=True)(net_speech)
    net_speech = Bidirectional(CuDNNLSTM(128,  
                               return_sequences=True))(net_speech)
    net_speech = Bidirectional(CuDNNLSTM(64,  
                               return_sequences=True))(net_speech)
    net_speech = Bidirectional(CuDNNLSTM(32,  
                               return_sequences=True))(net_speech)
    net_speech = Flatten()(net_speech)
    # net_speech = Dropout(0.3)(net_speech)

    model_combined = Dense(1)(net_speech)
    
    model = Model(input_speech, model_combined) 
    model.compile(loss=ccc_loss, 
                  optimizer='rmsprop', metrics=[ccc])
    return model


#def main(alpha, beta, gamma):
model = api_model()
model.summary()

# limiter first data of session 5 (for LOSO)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                          restore_best_weights=True)
hist = model.fit(feat[:limiter], vad[:limiter].T.tolist(), batch_size=200,
                 validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                 callbacks=[earlystop])
metrik = model.evaluate(feat[limiter:], vad[limiter:].T.tolist())
print(metrik)
# print('CCC_ave: ', np.mean(metrik[-3:]))