# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 04:22:18 2017

@author: Jashan
"""

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy


# random
import random

#################
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled
    
    ####
    sources = {'neg_txt.txt':'TRAIN_NEG', 'pos_text.txt':'TRAIN_POS', 'neu_text.txt':'TRAIN_NEU'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(10):
    model.train(sentences.sentences_perm(),total_examples=6870,epochs=20)
    
######### model.most_similar('जबरदस्त')
    
### TRAINING OUR NEURAL NET ON WORD EMBEDDINGS
    
train_arrays = numpy.zeros((6870, 100))
train_labels = numpy.zeros(6870)

for i in range(2290):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    prefix_train_neg = 'TRAIN_NEU_' + str(i)
    train_arrays[i] = model[prefix_train_pos]
    train_arrays[2290 + i] = model[prefix_train_neu]
    train_arrays[4580 + i] = model[prefix_train_neg]
    train_labels[i] = 1
    train_labels[2290 + i] = 1
    train_labels[4580 + i] = -1
    
    #########################################################################
###BUILDING NEURAL NET
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(train_arrays, train_labels,batch_size = 10, epochs = 100)
    