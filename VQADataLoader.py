#References - 
#http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from IPython.core.debugger import Pdb
from torchvision import transforms, utils
from PIL import Image

class VQADataSet(torch.utils.data.Dataset):
    def __init__(self,rootDir,train=False, validation=False,questionFileName=None,annotationFileName=None, imageDir=None, model_class=None, transforms=None, debug=False):
        #Tracer()()
        #Pdb().set_trace()
        self.rootDir = rootDir
        # Naming: q-question, wtoi- wordToIndex, itow- indexToWord dictionaries. atoi-answerToIndex.
        self.qwtoi, self.qitow = {}, {}
        self.atoi, self.itoa = {}, {}
        self.qwtoi['UNK'], self.atoi['UNK'] = len(self.qwtoi), len(self.atoi)
        self.qitow[0], self.itoa[0] = 'UNK', 'UNK'
        self.model_class = model_class
        self.transforms = transforms
        self.imageDir = imageDir
        self.debug = debug
        #
        quesfnT = os.path.join(rootDir,'v2_OpenEnded_mscoco_train2014_questions.json')
        annfnT = os.path.join(rootDir,'v2_mscoco_train2014_annotations.json')
        imgfnT = os.path.join(rootDir,'train2014')

        quesfnV = os.path.join(rootDir,'v2_OpenEnded_mscoco_val2014_questions.json')#FIXME @keshav: Must fill in
        annfnV = os.path.join(rootDir,'v2_mscoco_val2014_annotations.json')
        imgfnV = os.path.join(rootDir,'train2014')


        if train:
            qfn = quesfnT
            afn = annfnT
            idir = imgfnT
        elif validation:
            qfn = quesfnV
            afn = annfnV
            idir = imgfnV
        elif (os.path.exists(os.path.join(rootDir,questionFileName)) and
                os.path.exists(os.path.join(rootDir,imageDir))):
            qfn = os.path.join(rootDir,questionFileName)
            # afn = os.path.join(rootDir,annotationFileName)
            idir = os.path.join(rootDir,imageDir)
        else:
            print("neither train is True nor validation is True. Or any one of  questionFileName:{0} or imageDir {1}\
                missing".format(os.path.join(rootDir,questionFileName),os.path.join(rootDir,imageDir)))
            return

        #
        if questionFileName is None:
            #load train or validation set
            if(not os.path.exists(os.path.join(rootDir,'ques_vocab.pkl'))):
                training_data = self.getQuestionAnswerPairs(quesfnT, annfnT)

                self.populateAndWriteVocabs(training_data)

                if train:
                    self.data = training_data
                else:
                    validation_data = self.getQuestionAnswerPairs(quesfnV, annfnV)
                    self.data = validation_data
            #
            else:
                self.populateVocabsFromFile()
                self.data = self.getQuestionAnswerPairs(sfn,tfn)
        else:
            if (not os.path.exists(os.path.join(rootDir,'ques_vocab.pkl')) or not os.path.exists(os.path.join(rootDir,'ans_vocab.pkl'))):
                print("Vocab file missing.")
                return
            #
            self.populateVocabsFromFile()
            self.data = self.getQuestionAnswerPairs(sfn,tfn)
        #
        self.q_vocab_size = len(self.qwtoi)
        self.a_vocab_size = len(self.atoi)
        self.intData = []
        for qa_pair in self.data:
            q1 = self.prepare_sequence(qa_pair['question'],self.qwtoi)
            a1 = self.atoi[qa_pair['answer']]
            img_id = qa_pair['image_id']
            self.intData.append((q1,a1,img_id))
        #
        #random shuffle and sort by length of the string
        self.ind = np.arange(0,len(self.intData))
        if(train):
            np.random.seed(1)
            np.random.shuffle(self.ind)
            self.ind = list(self.ind)
            # self.ind.sort(key = lambda x: len(self.intData[x][0]))
        #
        # self.ix_to_tag = dict([[v,k] for k,v in self.tag_to_ix.items()])
        #Pdb().set_trace()
    

    def populateVocabsFromFile(self):
        self.qwtoi = pickle.load(open(os.path.join(self.rootDir,'ques_vocab.pkl'))) 
        self.qitow = pickle.load(open(os.path.join(self.rootDir,'ques_rev_vocab.pkl')))
        self.atoi = pickle.load(open(os.path.join(self.rootDir,'answer_vocab.pkl')))
        self.aiot = pickle.load(open(os.path.join(self.rootDir,'answer_rev_vocab.pkl')))
        #

    def populateAndWriteVocabs(self,data):

        for i, datum in enumerate(data):
            for word in datum['question']:
                if word not in self.qwtoi:
                    self.qwtoi[word] = len(self.qwtoi)
                    self.qitow[len(self.qwtoi)-1] = word

            answer = datum['answer']
            if answer not in self.atoi:
                self.atoi[answer] = len(self.atoi)
                self.itoa[len(self.atoi)-1] = answer

        pickle.dump(self.qwtoi, open(os.path.join(self.rootDir,'ques_vocab.pkl'),'w'))
        pickle.dump(self.atoi, open(os.path.join(self.rootDir,'ans_vocab.pkl'),'w'))
        pickle.dump(self.qitow,open(os.path.join(self.rootDir,'ques_rev_vocab.pkl'),'w'))
        pickle.dump(self.itoa,open(os.path.join(self.rootDir,'ans_rev_vocab.pkl'),'w'))


    def prepare_sequence(self,seq, to_ix):
        idxs = [to_ix.get(w,to_ix['UNK']) for w in seq]
        tensor = torch.LongTensor(idxs)
        return tensor
        #return autograd.Variable(tensor)


    def getQuestionAnswerPairs(self,qfn,afn):
        qf = json.load(open(qfn))
        af = json.load(open(afn))

        quess = qf['questions']
        anns = af['annotations'] 
        qas = []
        for i,ques in enumerate(quess):
            if(self.debug == True and i > 100):
                break
            qa = {}
            qa['question'] = ques['question']
            qa['question_id'] = ques['question_id']
            qa['image_id'] = ques['image_id']
            qa['answer'] = anns[i]['multiple_choice_answer']
            qas.append(qa)

        return qas


    def __len__(self):
        #Tracer()()
        return len(self.data)

    def __getitem__(self, idx):
        #Pdb().set_trace()
        question, answer, image_id = self.intData[self.ind[idx]]        
        image = None
        if(self.model_class == 'vqa'):
            if(os.path.exists('vqa_weights')):
                image = None

        elif(self.model_class == 'san'):
            if(os.path.exists('san_embs')):
                image = None

        if image is None:
            img = Image.open(os.path.join(self.rootDir,'train2014',"COCO_train2014_"+("%012d"%image_id)+'.jpg'))
            if self.transform is not None:
                img = self.transform(img)


        return question, answer, img