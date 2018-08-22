# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:06:08 2018

@author: sukandulapati
"""

import keras
from functools import reduce
import re
import numpy as np
import nltk
#nltk.download()
import json
from pprint import pprint as pp
from numpy import newaxis

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

def tokenize(sent):
    """Returns the tokens of a sequece"""
    tokens = nltk.word_tokenize(sent)
    tokens = [w.lower() for w in tokens]
    return tokens

with open('./SciQ dataset/train.json', 'r') as rf:
    train = json.load(rf)
with open('./SciQ dataset/test.json', 'r') as rf:
    test = json.load(rf)
with open('./SciQ dataset/valid.json', 'r') as rf:
    valid = json.load(rf)
    
import random
def preprocess(data_in):
    q = []
    s = []
    o = []
    l = []
    for sample in data_in:
        question = sample['question']
        support = sample['support']
        option1 = (sample['distractor1'], -1)
        option2 = (sample['distractor2'], -1)
        option3 = (sample['distractor3'], -1)
        option4 = (sample['correct_answer'], 1)
        options = [option1, option2, option3, option4]
        random.seed(1204)
        random.shuffle(options)
        q.append(question)
        s.append(support)
        o.append(tuple(op for op,_ in options))
        l.append(tuple(label for _l, label in options))
    X = {'questions': q, 'support': s, 'options': o}
    return X, l

def createVocab(input_data):
    vocab_list = set()
    for sample in input_data:
        s_t = tokenize(sample['support'])
        q_t = tokenize(sample['question'])
        d1_t = tokenize(sample['distractor1'])
        d2_t = tokenize(sample['distractor2'])
        d3_t = tokenize(sample['distractor3'])
        a_t = tokenize(sample['correct_answer'])
        vocab_list |= set(s_t+q_t+d1_t+d2_t+d3_t+a_t)
    vocab_list=sorted(vocab_list)
    vocab_size = len(vocab_list)+3
    vocab = dict((c,i+2) for i,c in enumerate(vocab_list))
    print("Vocab ready")
    return vocab_list, vocab_size, vocab

vocab_list, vocab_size, vocab = createVocab(train+valid+test)
    
def get_vectors(input_sent, vocab, vocab_list):
    tokenized = tokenize(input_sent)
    vectorized = []
    for w in tokenized:
        if w in vocab_list:
            vectorized.append(vocab[w])
        else:
            vectorized.append(vocab['UNK_ID'])
    return vectorized

def vectorize_input(X, y, vocab, vocab_size, support_maxlen, query_maxlen):
    op1 = []
    op2 = []
    op3 = []
    op4 = []
    l1 = []; l2 = []; l3 = []; l4 = []
    for label_list in y:
        l1.append(label_list[0])
        l2.append(label_list[1])
        l3.append(label_list[2])
        l4.append(label_list[3])
    labels = [np.array(l1),np.array(l2),np.array(l3),np.array(l4)]
    qs = [get_vectors(sent, vocab, vocab_list) for sent in X['questions']]
    sps = [get_vectors(sent, vocab, vocab_list) for sent in X['support']]
    for sample_options in X['options']:
        op1.append(get_vectors(sample_options[0], vocab, vocab_list))
        op2.append(get_vectors(sample_options[1], vocab, vocab_list))
        op3.append(get_vectors(sample_options[2], vocab, vocab_list))
        op4.append(get_vectors(sample_options[3], vocab, vocab_list))
    return(pad_sequences(qs, maxlen=query_maxlen),\
           pad_sequences(sps, maxlen=support_maxlen),\
           pad_sequences(op1, maxlen=query_maxlen),\
           pad_sequences(op2, maxlen=query_maxlen),\
           pad_sequences(op3, maxlen=query_maxlen),\
           pad_sequences(op4, maxlen=query_maxlen),\
           labels
          )
    
EMBED_SIZE = 300
Q_HIDDEN_SIZE = 100
S_HIDDEN_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 10
print('LSTM/EMBED/SUPPORT/QUERY={0},{1},{2},{3}'.format(LSTM,
                                                    EMBED_SIZE,
                                                    S_HIDDEN_SIZE,
                                                    Q_HIDDEN_SIZE))

support = layers.Input(shape=(S_HIDDEN_SIZE,), dtype='int32', name='support_input')
encoded_support = layers.Embedding(vocab_size, EMBED_SIZE)(support)
encoded_support = layers.Dropout(0.3)(encoded_support)
support_LSTM = LSTM(EMBED_SIZE)(encoded_support)
support_LSTM = layers.RepeatVector(S_HIDDEN_SIZE)(support_LSTM)

question = layers.Input(shape=(Q_HIDDEN_SIZE,), dtype='int32', name='question_input')
encoded_question = layers.Embedding(vocab_size, EMBED_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
question_LSTM = LSTM(EMBED_SIZE)(encoded_question)
question_LSTM = layers.RepeatVector(S_HIDDEN_SIZE)(question_LSTM)

MatchLSTM_layer = layers.add([support_LSTM, question_LSTM])
MatchLSTM_layer = layers.Dropout(0.5)(MatchLSTM_layer)
MatchLSTM_layer = LSTM(EMBED_SIZE, return_sequences=True)(MatchLSTM_layer)
MatchLSTM_layer = layers.Dense(S_HIDDEN_SIZE, activation = 'softmax')(MatchLSTM_layer)
print(MatchLSTM_layer)

def get_option_match(distractor, MatchLSTM_layer, option_num, S_HIDDEN_SIZE, Q_HIDDEN_SIZE):
    encoded_distractor = layers.Embedding(vocab_size, EMBED_SIZE)(distractor)
    encoded_distractor = layers.Dropout(0.3)(encoded_distractor)
    distractor_LSTM = LSTM(EMBED_SIZE)(encoded_distractor)
    added = layers.add([MatchLSTM_layer, distractor_LSTM])
    option_Match = layers.Dropout(0.3)(added)
    option_Match = LSTM(1)(option_Match)
    option_Match = layers.Dense(1, activation='tanh', name='op_{0}'.format(str(option_num)))(option_Match)
    return option_Match

distractor1 = layers.Input(shape=(Q_HIDDEN_SIZE,), dtype='int32', name='Option1_Input')
distractor2 = layers.Input(shape=(Q_HIDDEN_SIZE,), dtype='int32', name='Option2_Input')
distractor3 = layers.Input(shape=(Q_HIDDEN_SIZE,), dtype='int32', name='Option3_Input')
distractor4 = layers.Input(shape=(Q_HIDDEN_SIZE,), dtype='int32', name='Option4_Input')

option1_Match = get_option_match(distractor1,MatchLSTM_layer,1, S_HIDDEN_SIZE,Q_HIDDEN_SIZE)
option2_Match = get_option_match(distractor2,MatchLSTM_layer,2, S_HIDDEN_SIZE,Q_HIDDEN_SIZE)
option3_Match = get_option_match(distractor3,MatchLSTM_layer,3, S_HIDDEN_SIZE,Q_HIDDEN_SIZE)
option4_Match = get_option_match(distractor4,MatchLSTM_layer,4, S_HIDDEN_SIZE,Q_HIDDEN_SIZE)

model = Model([support, question, distractor1, distractor2, distractor3, distractor4], \
              [option1_Match, option2_Match, option3_Match, option4_Match])
print(model.summary())
#import pydot
#import graphviz
#plot_model(model, to_file='Final_Model.png')

model.compile(optimizer='adam',
             loss='squared_hinge',
             metrics=['accuracy'])

early_stop = EarlyStopping(monitor=['op_1_loss','op_2_loss','op_3_loss','op_4_loss'],
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')

print('Training')
X,y = preprocess(train)
q,s,d1,d2,d3, d4,a = vectorize_input(X, y, vocab, vocab_size, S_HIDDEN_SIZE, Q_HIDDEN_SIZE)

# checkpoint
filepath="\models\weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='op_1_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit([s,q, d1,d2,d3,d4], [a[0],a[1], a[2], a[3]],
         batch_size=128,
         epochs=50)

X_valid, y_valid = preprocess(valid)
vq,vs,vd1,vd2,vd3,vd4,va = vectorize_input(X_valid, y_valid, vocab, vocab_size, S_HIDDEN_SIZE, Q_HIDDEN_SIZE)
loss = model.evaluate([vs,vq, vd1,vd2,vd3,vd4], [va[0],va[1], va[2], va[3]],
                         batch_size=BATCH_SIZE)
# print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


p_op1,p_op2, p_op3, p_op4 = model.predict([vs,vq, vd1,vd2,vd3,vd4])

model.save("model_50_epochs_bs_128_new.h5")
#from keras.models import model_from_json
#model_json = model.to_json()
#with open("model_50_epochs_bs_128.json", "w") as wf:
    #wf.write(model_json)
    
# y_valid[:10]

# print(y[:32])

predictions = [[o1,o2,o3,o4] for o1,o2,o3,o4 in zip(p_op1,p_op2,p_op3,p_op4)]

#from Visualize import *

#print(highest_val_correct_acc(predictions,y_valid))

