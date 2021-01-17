### Roberta Model
##############################################

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
from transformers import RobertaTokenizer
import pickle
import re
import math
import numpy as np
import pandas as pd


train = pd.read_csv('../Data/train.csv')

##################################################
## define functions

def epoch_scheduler(epoch):
    return 3e-5 * 0.2**epoch

def load_weights(model, weight_model):
    with open(weight_model, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

def remove_html(text): ## remove html
    text = re.sub('https?://\S+|www\.\S+', '', text)
    return text

def save_weights(model, dst_fn):
    weights = model.get_weights()
    with open(dst_fn, 'wb') as f:
        pickle.dump(weights, f)

def loss_function(y_true, y_pred):
    # adjust the targets for sequence bucketing
    LEN = tf.shape(y_pred)[1]
    y_true = y_true[:, :LEN]
    size_batch = tf.shape(y_true)[0]
    ind_row = tf.range(0, LEN)
    ones_matrix = tf.ones([size_batch, LEN])
    #print(K.int_shape(ones_matrix))
    ind_row = tf.cast(ind_row ,dtype = tf.float32)
    ones_matrix = tf.cast(ones_matrix ,dtype = tf.float32)
    ind_matrix = ind_row * ones_matrix
    tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.float32)
    K.sum(y_pred * ind_matrix, axis=1)

    y_pred = tf.cast(y_pred ,dtype = tf.float32)
    y_true = tf.cast(y_true ,dtype = tf.float32)

    loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred,
        from_logits=False, label_smoothing=LABEL_SMOOTHING)
    loss2 = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
        from_logits=False, label_smoothing=LABEL_SMOOTHING)
    loss = tf.reduce_mean((loss1+loss2))
    return loss
### Define Model
def build_model():
    config = RobertaConfig.from_pretrained("roberta-base")
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')
    ## input layers for given its to roberta model
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)
    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    indices = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]
    ## Layers
    x = roberta_model(indices,attention_mask=att_,token_type_ids=tok_)

    x1 = tf.keras.layers.Dropout(0.23)(x[0])
    # This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.
    # If use_bias is True, a bias vector is created and added to the outputs.
    # Finally, if activation is not None, it is applied to the outputs as well
    x1 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Dropout(0.33)(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    # Finally, if activation is not None, it is applied to the outputs as well
    #x1 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    #Flatten: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('sigmoid')(x1)

    x2 = tf.keras.layers.Dropout(0.23)(x[0])
    x2 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Dropout(0.33)(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    # Finally, if activation is not None, it is applied to the outputs as well
    #x1 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x1)   
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('sigmoid')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss=loss_function ,optimizer=optimizer)

    # this is required as `model.predict` needs a fixed size!
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])
    return model, padded_model

def jaccard_similarity(text1, text2):
    a = set(text1.lower().split())
    b = set(text2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

MAX_LEN = 96 ## maximal lenth of a tweet
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
EPOCHS = 3 # originall_texty 3
size_batch = 32 # originall_texty 32
PAD_ID = 1
LABEL_SMOOTHING = 0.1
indice_sentiment = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
reduce_lr = tf.keras.callbacks.LearningRateScheduler(epoch_scheduler)

nb_lines = train.shape[0]
input_ids = np.ones((nb_lines,MAX_LEN),dtype='int32')
first_token = np.zeros((nb_lines,MAX_LEN),dtype='int32')
last_token = np.zeros((nb_lines,MAX_LEN),dtype='int32')
attention_mask = np.zeros((nb_lines,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((nb_lines,MAX_LEN),dtype='int32')

#train.isnull().sum()
#train.drop([314]) 27480
train.loc[314] = train.loc[27480]
#train.isnull().sum()
# drop all_text rows with any NaN and NaT values
#train2 = train.dropna()

for k in range(train.shape[0]):    

    print("iteration : ",k)
    text = " "+" ".join(train.loc[k,'text'].split())
    selected_text = " ".join(train.loc[k,'selected_text'].split())
    indice = text.find(selected_text)
    characters = np.zeros((len(text)))
    characters[indice:indice+len(selected_text)]=1

    if text[indice-1]==' ':
        characters[indice-1] = 1

    encoder = tokenizer.encode(text)
    print(encoder)    

    # ID_OFFSETS
    offsets = []; indice=0

    for t in encoder:
        w = tokenizer.decode([t])
        offsets.append((indice,indice+len(w)))
        indice += len(w)

    # START END TOKENS
    tokens = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(characters[a:b])
        if sm>0:
            tokens.append(i)

    sentiment_token = indice_sentiment[train.loc[k,'sentiment']]
    input_ids[k,:len(encoder)+3] = [0, sentiment_token] + encoder + [2]
    attention_mask[k,:len(encoder)+3] = 1

    if len(tokens)>0:
        first_token[k,tokens[0]+2] = 1
        last_token[k,tokens[-1]+2] = 1

    if k == 2:
        print(first_token[k])
        print(last_token[k])
        print(text)
        print(input_ids[k])
        print(selected_text)
        a = np.argmax(first_token[k])
        b = np.argmax(last_token[k])
        man = tokenizer.encode(text)
        print(tokenizer.decode(encoder[a-2:b-1]))

test = pd.read_csv('../Data/test.csv')


nb_lines = test.shape[0]

input_indicest = np.ones((nb_lines,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((nb_lines,MAX_LEN),dtype='int32')
token_type_indicest = np.zeros((nb_lines,MAX_LEN),dtype='int32')

for k in range(test.shape[0]):

    # INPUT_IDS
    text = " "+" ".join(test.loc[k,'text'].split())
    encoder = tokenizer.encode(text)
    sentiment_token = indice_sentiment[test.loc[k,'sentiment']]
    input_indicest[k,:len(encoder)+3] = [0, sentiment_token] + encoder + [2]
    attention_mask_t[k,:len(encoder)+3] = 1

## using StratifiedKFold
jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
start_of = np.zeros((input_ids.shape[0],MAX_LEN) )
end_of = np.zeros((input_ids.shape[0],MAX_LEN) )
predictions_start = np.zeros((input_indicest.shape[0],MAX_LEN) )
predictions_end = np.zeros((input_indicest.shape[0],MAX_LEN) )
skf = StratifiedKFold(n_splits=5,shuffle=True)#,random_state=SEED) #originall_texty 5 splits
for fold,(indiceT,indiceV) in enumerate(skf.split(input_ids,train.sentiment.values)):

    print('***'*25)
    print('** FOLD %i '%(fold+1))
    print('*'*25)

    ##################K.clear_session()
    model, padded_model = build_model()

    initial_input = [input_ids[indiceT,], attention_mask[indiceT,], token_type_ids[indiceT,]]
    target_test = [first_token[indiceT,], last_token[indiceT,]]
    input_validation = [input_ids[indiceV,],attention_mask[indiceV,],token_type_ids[indiceV,]]
    target_validation = [first_token[indiceV,], last_token[indiceV,]]

    # sort the validation data
    shuffle_validation = np.int32(sorted(range(len(input_validation[0])), key=lambda k: (input_validation[0][k] == PAD_ID).sum(), reverse=True))
    input_validation = [arr[shuffle_validation] for arr in input_validation]
    target_validation = [arr[shuffle_validation] for arr in target_validation]
    weight_model = '3-%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):
        # sort and shuffle: We add random numbers to not have the same order in each epoch
        shuffle_test = np.int32(sorted(range(len(initial_input[0])), key=lambda k: (initial_input[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))
        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch
        num_batches = math.ceil(len(shuffle_test) / size_batch)
        batch_inds = np.random.permutation(num_batches)
        shuffle_test_ = []
        for batch_ind in batch_inds:
            shuffle_test_.append(shuffle_test[batch_ind * size_batch: (batch_ind + 1) * size_batch])
        shuffle_test = np.concatenate(shuffle_test_)
        # reorder the input data
        initial_input = [arr[shuffle_test] for arr in initial_input]
        target_test = [arr[shuffle_test] for arr in target_test]
        model.fit(initial_input, target_test,
            epochs=epoch, initial_epoch=epoch - 1, batch_size=size_batch, verbose=DISPLAY,
            validation_data=(input_validation, target_validation), shuffle=False ,callbacks = [reduce_lr])  # don't shuffle in `fit`

        save_weights(model, weight_model)

    print('*** Loading model..........')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    load_weights(model, weight_model)

    print('*** Predicting.............')
    start_of[indiceV,],end_of[indiceV,] = padded_model.predict([input_ids[indiceV,],attention_mask[indiceV,],token_type_ids[indiceV,]],verbose=DISPLAY)


    # DISPLAY FOLD jaccard_similarity
    all_text = []
    for k in indiceV:
        a = np.argmax(start_of[k,])
        b = np.argmax(end_of[k,])
        if a>b:
            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            text = " "+" ".join(train.loc[k,'text'].split())
            encoder = tokenizer.encode(text)
            st = tokenizer.decode(encoder[a-2:b-1])
        all_text.append(jaccard_similarity(st,train.loc[k,'selected_text']))
    jac.append(np.mean(all_text))

    print('**** FOLD %i jaccard similarity ='%(fold+1),np.mean(all_text))
    print('Predicting Test...')

    predictions = padded_model.predict([input_indicest,attention_mask_t,token_type_indicest],verbose=DISPLAY)
    predictions_start += predictions[0]/skf.n_splits
    predictions_end += predictions[1]/skf.n_splits

##########################################################################
## submission file *******************************************************
test_input_id = np.ones((1 ,MAX_LEN))
test_attention_mask = np.zeros((1 ,MAX_LEN))
test_token_type_id = np.zeros((1 ,MAX_LEN))
test_input_id[0 ,:len(encoder)+3] = [0, sentiment_token] + encoder + [2]
attention_mask_t[0,:len(encoder)+3] = 1
start ,end = padded_model.predict([test_input_id ,test_attention_mask ,test_token_type_id])

all_text = []
for k in range(input_indicest.shape[0]):
    a = np.argmax(predictions_start[k,])
    b = np.argmax(predictions_end[k,])
    if a>b: 
        st = test.loc[k,'text']
    else:
        text = " "+" ".join(test.loc[k,'text'].split())
        encoder = tokenizer.encode(text)
        st = tokenizer.decode(encoder[a-2:b-1])
    all_text.append(st)

test['selected_text'] = all_text
test[['textID','selected_text']].to_csv('../Data/submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.sample(25)

    
