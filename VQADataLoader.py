#References - 
#http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

from IPython.core.debugger import Pdb

class POSDataSet(torch.utils.data.Dataset):
    def __init__(self,rootDir,train=False, validation=False,questionFileName=None,annotationFileName=None,imageFileName=None):
        #Tracer()()
        #Pdb().set_trace()
        self.rootDir = rootDir
        # Naming: q-question, a-answer, wtoi- wordToIndex, itow- indexToWord dictionaries
        self.qwtoi, self.qitow = {}, {}
        self.awtoi, self.aitow = {}, {}
        self.qwtoi['UNK'], self.awtoi['UNK'] = len(self.qwtoi), len(self.awtoi)
        self.qitow[0], self.aitow[0] = 'UNK', 'UNK'
        #
        quesfnT = os.path.join(rootDir,'v2_OpenEnded_mscoco_train2014_questions.json')
        annfnT = os.path.join(rootDir,'v2_mscoco_train2014_annotations.json')
        imgfnT = os.path.join(rootDir,'trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv')

        quesfnV = os.path.join(rootDir,'')#FIXME @keshav: Must fill in
        annfnV = os.path.join(rootDir,'')
        imgfnV = os.path.join(rootDir,'')

        if train:
            qfn = quesfnT
            afn = annfnT
            ifn = imgfnT
        elif validation:
            qfn = quesfnV
            afn = annfnV
            ifn = imgfnV
        elif (os.path.exists(os.path.join(rootDir,sentenceFileName)) and
                os.path.exists(os.path.join(rootDir,tagFileName))):
            qfn = os.path.join(rootDir,questionFileName)
            afn = os.path.join(rootDir,annotationFileName)
            ifn = os.path.join(rootDir,imageFileName)
        else:
            print("neither train is True nor validation is True. Or any one of  questionFileName:{0} or sentenceFileName {1} or imageFileName {2}\
                missing".format(os.path.join(rootDir,questionFileName),os.path.join(rootDir,annotationFileName),os.path.join(rootDir,imageFileName)))
            return
        #
        if sentenceFileName is None:
            #load train or validation set
            if(not os.path.exists(os.path.join(rootDir,'vocab.txt'))):
                qfn = json.load()
                training_data = self.getSentenceTagPairs(sfnS,tfnS)
                validation_data =self.getSentenceTagPairs(sfnV,tfnV)
                vocab_file = os.path.join(rootDir,'vocab.txt')
                tag_file = os.path.join(rootDir,'postags.txt')
                vf = open(vocab_file,'w+')
                tf = open(tag_file,'w+')
                self.populateAndWriteVocabsAndTags(training_data,vf,tf)
                self.populateAndWriteVocabsAndTags(validation_data,vf,tf)
                vf.close()
                tf.close()
                if train:
                    self.data = training_data
                else:
                    self.data = validation_data
            #
            else:
                self.populateVocabsAndTagsFromFile()
                self.data = self.getSentenceTagPairs(sfn,tfn)
        else:
            if (not os.path.exists(os.path.join(rootDir,'vocab.txt'))):
                print("Vocab file missing.")
                return
            #
            self.populateVocabsAndTagsFromFile()
            self.data = self.getSentenceTagPairs(sfn,tfn)
        #
        self.vocab_size = len(self.word_to_ix)
        self.intData = []
        for s,t in self.data:
            s1 = self.prepare_sequence(s,self.word_to_ix)
            t1 = self.prepare_sequence(t,self.tag_to_ix)
            self.intData.append((s1,t1))
        #
        #random shuffle and sort by length of the string
        self.ind = np.arange(0,len(self.intData))
        if(train):
            np.random.seed(1)
            np.random.shuffle(self.ind)
            self.ind = list(self.ind)
            self.ind.sort(key = lambda x: len(self.intData[x][0]))
        #
        self.ix_to_tag = dict([[v,k] for k,v in self.tag_to_ix.items()])
        #Pdb().set_trace()
    

    def populateVocabsAndTagsFromFile(self):
        vocab_file = os.path.join(self.rootDir,'vocab.txt')
        vf = open(vocab_file)
        vflines = vf.readlines()
        for word in vflines:
            self.word_to_ix[word[:-1]] = len(self.word_to_ix)
        #
        tag_file = os.path.join(self.rootDir,'postags.txt')
        tf = open(tag_file)
        tflines = tf.readlines()
        for tag in tflines:
            self.tag_to_ix[tag[:-1]] = len(self.tag_to_ix)
        #

    def populateAndWriteVocabsAndTags(self,data,vf,tf):
        for sent, tags in data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    print(word,file=vf)
            for tag in tags:
                if  tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    print(tag,file=tf)

    def prepare_sequence(self,seq, to_ix):
        idxs = [to_ix.get(w,to_ix['UNK']) for w in seq]
        tensor = torch.LongTensor(idxs)
        return tensor
        #return autograd.Variable(tensor)


    def getSentenceTagPairs(self,sfn,tfn):
        sfh = open(sfn)
        sentences = sfh.readlines()
        sentences = [x[:-1].split() for x in sentences]
        tfh = open(tfn)
        tags = tfh.readlines()
        tags = [x[:-1].split() for x in tags]
        training_data = zip(sentences,tags)
        return training_data

    def getQAPairs(self,qfn,afn):


    def __len__(self):
        #Tracer()()
        return len(self.data)

    def __getitem__(self, idx):
        #Pdb().set_trace()
        return self.intData[self.ind[idx]]
