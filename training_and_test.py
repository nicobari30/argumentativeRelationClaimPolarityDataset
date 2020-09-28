import tensorflow as tf
import keras as ker
import os
import string as s
from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import json
from networks import build_net_7_nc
from keras.optimizers import Adam
from training_utils import get_avgF1
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tabulate import tabulate

########## Lettura dataset ##########

dataset = pd.read_csv('claim_stance_dataset_v1.csv')

########## Creazione dataset per classificazione ##########




dataset = dataset[dataset["claims.Compatible"] == 'yes']

train_data = dataset[dataset.split == 'train']
test_data = dataset[dataset.split == 'test']

x = train_data[['topicText', 'claims.claimCorrectedText']]
y = train_data[['topicSentiment', 'claims.claimSentiment', 'claims.stance', 'claims.targetsRelation']]

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


########## Tokenization ##########

t = ker.preprocessing.text.Tokenizer()
to_fit=X_train['topicText']+" "+X_train['claims.claimCorrectedText']

t.fit_on_texts(to_fit)




word_index = t.word_index
print('Found %s unique tokens.' % len(word_index))
dict_num = len(word_index) + 1

max_len = 0

for topic in dataset['topicText']:
    max_len=max(len(t.texts_to_sequences(topic)),max_len)
    
for claim in dataset['claims.claimCorrectedText']:
    max_len=max(len(t.texts_to_sequences(claim)),max_len)


topic_train = t.texts_to_sequences(X_train['topicText'])
topic_train_padded = ker.preprocessing.sequence.pad_sequences(topic_train, maxlen=max_len)

claim_train = t.texts_to_sequences(X_train['claims.claimCorrectedText'])
claim_train_padded = ker.preprocessing.sequence.pad_sequences(claim_train, maxlen=max_len)

topic_test = t.texts_to_sequences(test_data['topicText'])
topic_test_padded = ker.preprocessing.sequence.pad_sequences(topic_test, maxlen=max_len)

claim_test = t.texts_to_sequences(test_data['claims.claimCorrectedText'])
claim_test_padded = ker.preprocessing.sequence.pad_sequences(claim_test, maxlen=max_len)

topic_val= t.texts_to_sequences(X_val['topicText'])
topic_val_padded= ker.preprocessing.sequence.pad_sequences(topic_val,maxlen=max_len)

claim_val= t.texts_to_sequences(X_val['claims.claimCorrectedText'])
claim_val_padded= ker.preprocessing.sequence.pad_sequences(claim_val,maxlen=max_len)

stance_test = []
stance_train = []
stance_val=[]

targets_relation_train=y_train['claims.targetsRelation']
topic_target_sentiment_train=y_train['topicSentiment']
claim_target_sentiment_train=y_train['claims.claimSentiment']

targets_relation_test=test_data['claims.targetsRelation']
claim_target_sentiment_test=test_data['claims.claimSentiment']
topic_target_sentiment_test=test_data['topicSentiment']

targets_relation_val=y_val['claims.targetsRelation']
claim_target_sentiment_val=y_val['claims.claimSentiment']
topic_target_sentiment_val=y_val['topicSentiment']


for stance in y_train['claims.stance']:
    if stance == "PRO":
        stance_train.append(1)
    else:
        stance_train.append(0)

for stance in test_data['claims.stance']:
    if stance == "PRO":
        stance_test.append(1)
    else:
        stance_test.append(-1)

for stance in y_val['claims.stance']:
    if stance == "PRO":
        stance_val.append(1)
    else:
        stance_val.append(0)

stance_weights = class_weight.compute_class_weight('balanced',np.unique(stance_train),stance_train)
topic_weights = class_weight.compute_class_weight('balanced',np.unique(topic_target_sentiment_train),topic_target_sentiment_train)
claim_weights = class_weight.compute_class_weight('balanced',np.unique(claim_target_sentiment_train),claim_target_sentiment_train)
rel_weights = class_weight.compute_class_weight('balanced',np.unique(targets_relation_train),targets_relation_train)

class_weights = {0:{0: topic_weights[0], 1: topic_weights[1]},1:{0: claim_weights[0], 1:claim_weights[1]},2:{0: stance_weights[0],1:stance_weights[1]},3:{0:rel_weights[0],1:rel_weights[1]}}


print(class_weights)

num_claim_train, max_wclaim_train = claim_train_padded.shape
num_claim_test, max_wclaim_test = claim_test_padded.shape
num_topic_train, max_wtopic_train = topic_train_padded.shape
num_topic_test, max_wtopic_test = topic_test_padded.shape

# Trasformo gli array
targets_relation_train = np.where(targets_relation_train==1,targets_relation_train,0)
targets_relation_train=ker.utils.to_categorical(targets_relation_train,num_classes=2)
claim_target_sentiment_train=np.where(claim_target_sentiment_train==1,claim_target_sentiment_train,0)
claim_target_sentiment_train=ker.utils.to_categorical(claim_target_sentiment_train,num_classes=2)
topic_target_sentiment_train = np.where(topic_target_sentiment_train==1,topic_target_sentiment_train,0)
topic_target_sentiment_train=ker.utils.to_categorical(topic_target_sentiment_train,num_classes=2)
stance_train = ker.utils.to_categorical(stance_train, num_classes=2)

targets_relation_val= np.where(targets_relation_val==1,targets_relation_val,0)
targets_relation_val = ker.utils.to_categorical(targets_relation_val, num_classes=2)
claim_target_sentiment_val=np.where(claim_target_sentiment_val==1,claim_target_sentiment_val,0)
claim_target_sentiment_val = ker.utils.to_categorical(claim_target_sentiment_val, num_classes=2)
topic_target_sentiment_val = np.where(topic_target_sentiment_val==1,topic_target_sentiment_val,0)
topic_target_sentiment_val = ker.utils.to_categorical(topic_target_sentiment_val, num_classes=2)
stance_val = ker.utils.to_categorical(stance_val, num_classes=2)

########## Definizione modello neurale ##########


# Utilizzo la rete


embeddings_index = {}
f = open(os.path.join(".", 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((dict_num, 100))
for word, i in t.word_index.items():
    embedding_value = embeddings_index.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

model = build_net_7_nc(embedding_weights=embedding_matrix,
                       text_max_length=max_wclaim_train,
                       embedding_dimension=100,
                       distance=0,
                       share_embedder=True)

model.compile(optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics={
                  "topic_target_sentiment": [get_avgF1([1])],
                  "claim_target_sentiment": [get_avgF1([1])],
                  "stance_relation": [get_avgF1([1])],
                  "targets_relation": [get_avgF1([1])],
              },
              loss_weights=[1, 1, 1, 1])
model.summary()

monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto', restore_best_weights=True)

X_train = [claim_train_padded, topic_train_padded]
y_train = [topic_target_sentiment_train, claim_target_sentiment_train, stance_train, targets_relation_train]

X_val = [claim_val_padded, topic_val_padded]
y_val = [topic_target_sentiment_val, claim_target_sentiment_val, stance_val, targets_relation_val]





history = model.fit(X_train, y_train, validation_data=(X_val,y_val), callbacks=[monitor], epochs=80, batch_size=32, class_weight=class_weights)





preds = model.predict(x=[claim_test_padded, topic_test_padded])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['claim_target_sentiment_F1_1'])
plt.plot(history.history['val_claim_target_sentiment_F1_1'])
plt.title('claim target sentiment f1')
plt.ylabel('f1 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['topic_target_sentiment_F1_1'])
plt.plot(history.history['val_topic_target_sentiment_F1_1'])
plt.title('topic target sentiment f1')
plt.ylabel('f1 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['stance_relation_F1_1'])
plt.plot(history.history['val_stance_relation_F1_1'])
plt.title('stance relation f1')
plt.ylabel('f1 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['targets_relation_F1_1'])
plt.plot(history.history['val_targets_relation_F1_1'])
plt.title('target relation F1')
plt.ylabel('f1 score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

topic_target_res=[]
claim_target_res=[]
stance_res=[]
target_relation_res=[]




for el in preds[0]:
    if el[0]>el[1]:
        topic_target_res.append(-1)
    else:
        topic_target_res.append(1)

for el in preds[1]:
    if el[0]>el[1]:
        claim_target_res.append(-1)
    else:
        claim_target_res.append(1)
        
for el in preds[2]:
    if el[0]>el[1]:
        stance_res.append(-1)
    else:
        stance_res.append(1)


for el in preds[3]:
    if el[0]>el[1]:
        target_relation_res.append(-1)
    else:
        target_relation_res.append(1)


alfa=np.arange(0., 1.5, 0.05)
coverage=np.arange(0.1,1.1,0.1)
max_for_a=0
res=[]
for min_coverage in coverage:
    valid=[]
    for a in alfa:
        gt=[]
        for p in preds[2]:
            if (p[0]>=a) or (p[1]>=a):
                gt.append(p)
        if (len(gt)/len(preds[2]))>= min_coverage:
            valid.append(a)
    for a_valid in valid:
        acc=[]
        gt_valid=[]
        for p in preds[2]:
            if (p[0]>=a_valid) or (p[1]>=a_valid):
                gt_valid.append(p)
        counter=0
        for p in preds[2]:
            if (p[0]>p[1] and p[0]>=a_valid and stance_test[counter]==0) or (p[1]>p[0] and p[1]>=a_valid and stance_test[counter]==1):
                acc.append(p)
            counter+=1
        if (max_for_a<=(len(acc)/len(gt_valid))):
                max_for_a=(len(acc)/len(gt_valid))
    res.append(max_for_a)
    max_for_a=0
            

print("////TOPIC TARGET METRICS////")
topic_tar_accuracy=accuracy_score(topic_target_sentiment_test,topic_target_res)
print("Accuracy: "+ str(accuracy_score(topic_target_sentiment_test,topic_target_res)))
topic_tar_f1=f1_score(topic_target_sentiment_test,topic_target_res)
print("F1_score: "+ str(f1_score(topic_target_sentiment_test,topic_target_res)))
topic_tar_prec=precision_score(topic_target_sentiment_test,topic_target_res)
print("Precision: "+ str(precision_score(topic_target_sentiment_test,topic_target_res)))
topic_tar_rec=recall_score(topic_target_sentiment_test,topic_target_res)
print("Recall: "+ str(recall_score(topic_target_sentiment_test,topic_target_res)))

print("////CLAIM TARGET METRICS////")
claim_tar_accuracy=accuracy_score(claim_target_sentiment_test,claim_target_res)
print("Accuracy: "+ str(accuracy_score(claim_target_sentiment_test,claim_target_res)))
claim_tar_f1=f1_score(claim_target_sentiment_test,claim_target_res)
print("F1_score: "+ str(f1_score(claim_target_sentiment_test,claim_target_res)))
claim_tar_prec=precision_score(claim_target_sentiment_test,claim_target_res)
print("Precision: "+ str(precision_score(claim_target_sentiment_test,claim_target_res)))
claim_tar_rec=recall_score(claim_target_sentiment_test,claim_target_res)
print("Recall: "+ str(recall_score(claim_target_sentiment_test,claim_target_res)))
                    
print("////STANCE METRICS////")
stance_acc=accuracy_score(stance_test,stance_res)
print("Accuracy: "+ str(accuracy_score(stance_test,stance_res)))
stance_f1=f1_score(stance_test,stance_res)
print("F1_score: "+ str(f1_score(stance_test,stance_res)))
stance_prec=precision_score(stance_test,stance_res)
print("Precision: "+ str(precision_score(stance_test,stance_res)))
stance_rec=recall_score(stance_test,stance_res)
print("Recall: "+ str(recall_score(stance_test,stance_res)))

print("////TARGET RELATION METRICS////")
tr_acc=accuracy_score(targets_relation_test,target_relation_res)
print("Accuracy: "+ str(accuracy_score(targets_relation_test,target_relation_res)))
tr_f1=f1_score(targets_relation_test,target_relation_res)
print("F1_score: "+ str(f1_score(targets_relation_test,target_relation_res)))
tr_prec=precision_score(targets_relation_test,target_relation_res)
print("Precision: "+ str(precision_score(targets_relation_test,target_relation_res)))
tr_rec=recall_score(targets_relation_test,target_relation_res)
print("Recall: "+ str(recall_score(targets_relation_test,target_relation_res)))


stance = pd.DataFrame ({'0.1':[res[0]],
                        '0.2':[res[1]],
                        '0.3':[res[2]],
                        '0.4':[res[3]],
                        '0.5':[res[4]],
                        '0.6':[res[5]],
                        '0.7':[res[6]],
                        '0.8':[res[7]],
                        '0.9':[res[8]],
                        '1.0':[res[9]]})
                               

metrics = pd.DataFrame({' ':["Accuracy","F1_Score","Precision","Recall"],
                        'Topic Target Sentiment':[round(topic_tar_accuracy,2),round(topic_tar_f1,2),round(topic_tar_prec,2),round(topic_tar_rec,2)],
                        'Claim Target Sentiment':[round(claim_tar_accuracy,2),round(claim_tar_f1,2),round(claim_tar_prec,2),round(claim_tar_rec,2)],
                        'Stance':[round(stance_acc,2),round(stance_f1,2),round(stance_prec,2),round(stance_rec,2)],
                        'Targets Relation':[round(tr_acc,2),round(tr_f1,2),round(tr_prec,2),round(tr_rec,2)]
                        })
print(tabulate(stance, headers='keys', tablefmt='psql'))
print(tabulate(metrics, headers='keys', tablefmt='psql'))



    
            
