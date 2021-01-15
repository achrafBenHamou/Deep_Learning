##
##############################################S
import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
from transformers import RobertaTokenizer
import tokenizers
import pickle
import re
import math
import numpy as np
import pandas as pd
#from google.colab import files
print('TF version',tf.__version__)

train = pd.read_csv('./Data/train.csv')

## maximal lenth of a tweet

MAX_LEN = 96

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

EPOCHS = 3 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

LABEL_SMOOTHING = 0.1

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

## remove html 

def remove_html(text):

    text = re.sub('https?://\S+|www\.\S+', '', text)

    return text

def scheduler(epoch):

    return 3e-5 * 0.2**epoch

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

train

ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

train.shape[0]

train.isnull().sum()

#train.drop([314]) 27480

train.loc[314] = train.loc[27480] 

train.isnull().sum()

# drop all rows with any NaN and NaT values

train2 = train.dropna()

text1 = " "+" ".join(train2.loc[314,'text'].split())

text1

for k in range(train.shape[0]):    

    # FIND OVERLAP

    print("iteration ",k)

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

    print(enc)    

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc)+3] = [0, s_tok] + enc + [2]

    attention_mask[k,:len(enc)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1

    if k == 2:

        print(start_tokens[k])

        print(end_tokens[k])

        print(text1)

        print(input_ids[k])

        print(text2)

        a = np.argmax(start_tokens[k])

        b = np.argmax(end_tokens[k])

        man = tokenizer.encode(text1)

        print(tokenizer.decode(enc[a-2:b-1]))

test = pd.read_csv('./Data/test.csv')



ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc)+3] = [0, s_tok] + enc + [2]

    attention_mask_t[k,:len(enc)+3] = 1



def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model

def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)

def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    batch_size = tf.shape(y_true)[0]

    LEN = ll 

    ind_row = tf.range(0, LEN)

    ones_matrix = tf.ones([batch_size, LEN])

    #print(K.int_shape(ones_matrix))

    ind_row = tf.cast(ind_row ,dtype = tf.float32)

    ones_matrix = tf.cast(ones_matrix ,dtype = tf.float32)     

    ind_matrix = ind_row * ones_matrix

    k1 = tf.cast(tf.math.argmax(y_true ,axis = 1) ,dtype = tf.float32)

    k2 = K.sum(y_pred * ind_matrix, axis=1)

    y_pred = tf.cast(y_pred ,dtype = tf.float32) 

    y_true = tf.cast(y_true ,dtype = tf.float32) 

    loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss2 = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean((loss1+loss2))

    return loss

def build_model():

    ##########config = RobertaConfig.from_pretrained('../input/tf-roberta/config-roberta-base.json')

    config = RobertaConfig.from_pretrained("roberta-base")

    ####bert_model = TFRobertaModel.from_pretrained('../input/tf-roberta/pretrained-roberta-base.h5',config=config)

    bert_model = TFRobertaModel.from_pretrained('roberta-base')

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    x1 = tf.keras.layers.Dropout(0.15)(x[0])

    x1 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.15)(x[0]) 

    x2 = tf.keras.layers.Conv1D(1536, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(128, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)



    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn ,optimizer=optimizer)

    

    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))

import tensorflow.keras.backend as K

#from transformers import RobertaConfig, RobertaModel, TFRobertaModel

import math

jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN) )

oof_end = np.zeros((input_ids.shape[0],MAX_LEN) )

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN) )

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN) )

skf = StratifiedKFold(n_splits=5,shuffle=True)#,random_state=SEED) #originally 5 splits

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    ##################K.clear_session()

    model, padded_model = build_model()

        

    #sv = tf.keras.callbacks.ModelCheckpoint(

    #    '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

    #    save_weights_only=True, mode='auto', save_freq='epoch')

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '3-%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY,

            validation_data=(inpV, targetV), shuffle=False ,callbacks = [reduce_lr])  # don't shuffle in `fit`

        save_weights(model, weight_fn)



    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            ##error

            ##st = tokenizer.decode(enc.ids[a-2:b-1])

            st = tokenizer.decode(enc[a-2:b-1])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    print()

################################################################
test_input_id = np.ones((1 ,MAX_LEN))
test_attention_mask = np.zeros((1 ,MAX_LEN))
test_token_type_id = np.zeros((1 ,MAX_LEN))
sentence =  'I am very much delighted'
sent = 'positive'
sentence = ' '+' '.join(sentence.split())
enc = tokenizer.encode(sentence)              
s_tok = sentiment_id[sent]
test_input_id[0 ,:len(enc)+3] = [0, s_tok] + enc + [2]
attention_mask_t[0,:len(enc)+3] = 1
start ,end = padded_model.predict([test_input_id ,test_attention_mask ,test_token_type_id])
a = np.argmax(start[0 ,])
b = np.argmax(end[0 ,])
if a>b:
    print(sentence)
else:
    selected = tokenizer.decode(enc[a-2:b-1])
    print(selected)


all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test.loc[k,'text']
    else:
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc[a-2:b-1])
    all.append(st)


test['selected_text'] = all
test[['textID','selected_text']].to_csv('.Data/submission.csv',index=False)
pd.set_option('max_colwidth', 60)
test.sample(25)

    
