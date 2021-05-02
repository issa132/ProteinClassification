# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:46:23 2021

@author: BABAN CHAWAI
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plot #plt
import seaborn as sns
import keras
sns.set_context("notebook", font_scale=1.4)


from collections import Counter


from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout #Flatten, Activation
from keras.layers import Embedding, Bidirectional, LSTM #GlobalMaxPooling1D



import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 

data_path = 'C:\\Users\\BABAN CHAWAI\\Desktop\\archive\\random_split\\random_split'
print('Available data', os.listdir(data_path))


# lecture et concaténation des données pour chaque dossier.
def read_data(split):
  data = []
  for fn in os.listdir(os.path.join(data_path, split)):
    with open(os.path.join(data_path, split, fn)) as f:
      data.append(pd.read_csv(f, index_col=None))
  return pd.concat(data)


# lecture de toutes les partitions de données
df_train = read_data('train')
df_val = read_data('val')
df_test = read_data('test')

df_train.info()
df_train.head()
df_train.head(1)['sequence'].values[0]

print('Volume données d''entrainement:  ', len(df_train))
print('Volume données de validation: ', len(df_val))
print('Volume données Test: ', len(df_test))

def classes_unique(train, test, val):

  train_unq = np.unique(train['family_accession'].values)
  val_unq = np.unique(val['family_accession'].values)
  test_unq = np.unique(test['family_accession'].values)

  print('Nombre de classes uniques dans les données Entrainement: ', len(train_unq))
  print('Nombre de classes uniques dans les données de Validation: ', len(val_unq))
  print('Nombre de classes uniques dans les données de Test: ', len(test_unq))
  
classes_unique(df_train, df_test, df_val)
  
  
  # Longueur de séquence dans les données d'entrainement.
df_train['seq_char_count']= df_train['sequence'].apply(lambda x: len(x))
df_val['seq_char_count']= df_val['sequence'].apply(lambda x: len(x))
df_test['seq_char_count']= df_test['sequence'].apply(lambda x: len(x))

def plot_seq_count(df, data_name):
  sns.histplot(df['seq_char_count'].values,  color="red", label="100% Equities")
  #plt.title('Sequence char count: {data_name}')
  plot.title('Sequence char count: {}'.format(data_name))
  
  plot.grid(True)

plot.subplot(1, 3, 1)
plot_seq_count(df_train, 'Train')

plot.subplot(1, 3, 2)
plot_seq_count(df_val, 'Val')

plot.subplot(1, 3, 3)
plot_seq_count(df_test, 'Test')

plot.subplots_adjust(right=3.0)
plot.show()

#Fréquence des codes dans chaque séquence

def freq_codes_seq(df, data_name):
  
  df = df.apply(lambda x: " ".join(x))
  
  codes = []
  # concaténation des differents codes (Lettres) présents dans la séquence
  for i in df: # concatination of all codes
    codes.extend(i)

  codes_dict= Counter(codes)
  #suppression d'espace
  codes_dict.pop(' ')  
  
 
  print('Codes: {}'.format(data_name))
  print('Nombre Total de codes uniques: {}'.format(len(codes_dict.keys())))
  

 
  df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
  return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]
  

#  La frequence des codes dans les sequences de données d'entrainement
train_code_freq = freq_codes_seq(df_train['sequence'], 'Train')
train_code_freq

# si on met ceci dans la console ça donne plus de details "train_code_freq"


 # La frequence des codes dans les sequences de données de validation
val_code_freq = freq_codes_seq(df_val['sequence'], 'Val')
val_code_freq


# La frequence des codes dans les sequences de données Test
test_code_freq = freq_codes_seq(df_test['sequence'], 'Test')
test_code_freq


def plot_code_freq(df, data_name):
  
  plot.title(' frequence des codes (les lettres): {}'.format(data_name))
  sns.barplot(x='Code', y='Freq', data=df)

plot.subplot(1, 3, 1)
plot_code_freq(train_code_freq, 'Train')

plot.subplot(1, 3, 2)
plot_code_freq(val_code_freq, 'Val')

plot.subplot(1, 3, 3)
plot_code_freq(test_code_freq, 'Test')

plot.subplots_adjust(right=3.0)
plot.show()


classes = df_train['family_accession'].value_counts()[:1000].index.tolist()
len(classes)


#Using the loc() function, we can access the data values fitted in the particular row or column based on the index value passed to the function.

train_sm = df_train.loc[df_train['family_accession'].isin(classes)].reset_index()
val_sm = df_val.loc[df_val['family_accession'].isin(classes)].reset_index()
test_sm = df_test.loc[df_test['family_accession'].isin(classes)].reset_index()

#print('Data size after considering 1000 classes for each data split:')
#print('Train size :', len(train_sm))
#print('Val size :', len(val_sm))
#print('Test size :', len(test_sm))

classes_unique(train_sm, test_sm, val_sm)

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)

print(char_dict)
print("La longueur du dictionnaire:", len(char_dict))

def integer_encoding(data):
  #Permet de coder les differents accides aminées présents dans la sequence
  encode_list = []
  for row in data['sequence'].values:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list





train_encode = integer_encoding(train_sm) 
val_encode = integer_encoding(val_sm) 
test_encode = integer_encoding(test_sm)

# padding sequences

max_length = 100
train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

train_pad.shape, val_pad.shape, test_pad.shape

# One hot encoding of sequences

train_ohe = to_categorical(train_pad)
val_ohe = to_categorical(val_pad)
test_ohe = to_categorical(test_pad)

train_ohe.shape, test_ohe.shape, test_ohe.shape


plot.style.use('ggplot')

def plot_history(history):
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history1['val_loss']
  x = range(1, len(acc) + 1)

  plot.figure(figsize=(12, 5))
  plot.subplot(1, 2, 1)
  plot.plot(x, acc, 'b', label='Précision pour l''entrainement')
  plot.plot(x, val_acc, 'r', label='Précision pour la validation')
  plot.title('Précision pour l''entrainement et pour la validation')
  plot.legend()

  plot.subplot(1, 2, 2)
  plot.plot(x, loss, 'b', label=' loss pour l''entrainement')
  plot.plot(x, val_loss, 'r', label=' loss pour la Validation')
  plot.title('La valeur du loss pour les données de validation et d''entrainement')
  plot.legend()
  
def display_model_score(model, train, val, test, batch_size):

  train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
  print('Perte pour les données d''entrainement: ', train_score[0])
  print('Précision(accuracy) pour les données d''entrainement : ', train_score[1])
  print('-'*100)

  val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
  print('Perte(loss) pour les données de validation: ', val_score[0])
  print('Précision(accuracy) pour les données de Validation: ', val_score[1])
  print('-'*100)
  
  test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
  print('Perte pour les données Test: ', test_score[0])
  print('Précision(accuracy) pour les données Test: ', test_score[1])  


x_input = Input(shape=(100,))
emb = Embedding(21, 128, input_length=max_length)(x_input)
bi_rnn = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
x = Dropout(0.3)(bi_rnn)

# classificateur softmax
x_output = Dense(1000, activation='softmax')(x)
model1 = keras.models.Sequential()
model1 = Model(inputs=x_input, outputs=x_output)
optimizer = keras.optimizers.RMSprop(lr=0.01)
model1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()



x_input = Input(shape=(100,))
emb = Embedding(21, 128, input_length=max_length)(x_input)
bi_rnn = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
x = Dropout(0.3)(bi_rnn)

# softmax classifier
x_output = Dense(1000, activation='softmax')(x)

model1 = Model(inputs=x_input, outputs=x_output)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()


# Training will stop when the chosen performance measure stops improving.
# you want to minimize loss or maximize accuracy.
# with mode = min, training will stop when the quantity monitored has stopped decreasing
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# patience: Number of epochs with no improvement after which training will be stopped.

history1 = model1.fit(
    train_pad, y_train,
    epochs=50, batch_size=256,
    validation_data=(val_pad, y_val),
    #callbacks = [es]
    callbacks= tf.keras.callbacks.EarlyStopping(monitor='val_loss', mpatience=3, verbose=1)
    
    )

 # saving model weights.
model1.save_weights('C:\\Users\\BABAN CHAWAI\\weights\\model2.h5')


plot_history(history1)


display_model_score(model1,
    [train_pad, y_train],
    [val_pad, y_val],
    [test_pad, y_test],
    256)


