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
import UtilityClasses as uc
from collections import Counter
#TODO -  take care of missing keys UNK while reading from dictionaries
#done
#TODO - make wtoi and itow dictionaries static so that same file is not loaded multiple times
#done


class VQADataSet(torch.utils.data.Dataset):
    #Making them class variables so that vocabs not loaded again and again
    qwtoi = None
    qitow = None
    atoi = None
    itoa = None

    def __init__(self,rootDir,train=False, validation=False,questionFileName=None,annotationFileName=None, imageDir=None, model_class=None, transforms=None, debug=False,max_answers=1000,batch_size = 1,features_dir=None):
        #Tracer()()
        self.features_dir = features_dir
        self.max_answers = max_answers
        self.batch_size = batch_size
        #Pdb().set_trace()
        self.rootDir = rootDir
        # Naming: q-question, wtoi- wordToIndex, itow- indexToWord dictionaries. atoi-answerToIndex.
        #
        self.model_class = model_class
        self.transforms = transforms
        self.imageDir = imageDir
        self.debug = debug
        #
        quesfnT = os.path.join(rootDir,'v2_OpenEnded_mscoco_train2014_questions.json')
        annfnT = os.path.join(rootDir,'v2_mscoco_train2014_annotations.json')
        imgfnT = os.path.join(rootDir,'train2014/COCO_train2014_')

        quesfnV = os.path.join(rootDir,'v2_OpenEnded_mscoco_val2014_questions.json')
        annfnV = os.path.join(rootDir,'v2_mscoco_val2014_annotations.json')
        imgfnV = os.path.join(rootDir,'val2014/COCO_val2014_')


        if train:
            qfn = quesfnT
            afn = annfnT
            self.idir = imgfnT
        elif validation:
            qfn = quesfnV
            afn = annfnV
            self.idir = imgfnV
        elif (os.path.exists(os.path.join(rootDir,questionFileName)) and
                os.path.exists(os.path.join(rootDir,imageDir))):
            qfn = os.path.join(rootDir,questionFileName)
            afn = os.path.join(rootDir,annotationFileName)
            self.idir = os.path.join(rootDir,imageDir)
        else:
            print("neither train is True nor validation is True. Or any one of  questionFileName:{0} or imageDir {1}\
                missing".format(os.path.join(rootDir,questionFileName),os.path.join(rootDir,imageDir)))
            return

        #
        if questionFileName is None:
            #load train or validation set
            if(
            (not os.path.exists(os.path.join(rootDir,'ques_vocab.pkl'))) or
            (not os.path.exists(os.path.join(rootDir,'ques_rev_vocab.pkl'))) or
            (not os.path.exists(os.path.join(rootDir,'ans_vocab_'+str(self.max_answers)+'.pkl'))) or
            (not os.path.exists(os.path.join(rootDir,'ans_rev_vocab_'+str(self.max_answers)+'.pkl')))
             ):
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
                self.data = self.getQuestionAnswerPairs(qfn,afn)
        else:
            if ((not os.path.exists(os.path.join(rootDir,'ques_vocab.pkl'))) or (not os.path.exists(os.path.join(rootDir,'ans_vocab_'+str(self.max_answers)+'.pkl')))):
                print("Vocab file missing.")
                return
            #
            self.populateVocabsFromFile()
            self.data = self.getQuestionAnswerPairs(qfn,afn)
        #
        self.q_vocab_size = len(VQADataSet.qwtoi)
        self.a_vocab_size = len(VQADataSet.atoi)
        self.intData = []
        self.populateIntData(qfn)
        #
        #random shuffle and sort by length of the string
        self.ind = np.arange(0,len(self.data))
        if(train):
            #Pdb().set_trace()
            np.random.seed(1)
            np.random.shuffle(self.ind)
            self.ind = list(self.ind)
            if self.batch_size > 1:
                self.ind.sort(key = lambda x: len(self.data[x]['question']))
                N = len(self.ind)
                if N > 10:
                    self.block_ids = {}
                    random_block_ids = list(range(N))
                    np.random.shuffle(random_block_ids)
                    #generate a random number between 0 to N - 1
                    blockid = random_block_ids[0]
                    self.block_ids[self.ind[0]] = blockid
                    running_count = 1
                    for ind_it in range(1,N):
                        if running_count >= self.batch_size or len(self.data[self.ind[ind_it]]['question']) != len(self.data[self.ind[ind_it-1]]['question']):
                            blockid = random_block_ids[ind_it]
                            running_count = 0
                        #
                        self.block_ids[self.ind[ind_it]] = blockid
                        running_count += 1
                    #
                    self.ind.sort(key = lambda x: self.block_ids[x])
                    #self.ind  = [x for _,x in sorted(zip(self.block_ids,self.ind))]
                    #self.ind.sort(key= lambda x: self.block_ids[x])

                #generate
        #
        # self.ix_to_tag = dict([[v,k] for k,v in self.tag_to_ix.items()])
        #Pdb().set_trace()


    def populateIntData(self,qfn):
        basedir = os.path.dirname(qfn)
        basename_qfn = os.path.basename(qfn)
        pickle_path = os.path.join(basedir,basename_qfn+'_intData.pkl')
        if os.path.exists(pickle_path):
            print('unpickle intData')
            self.intData = pickle.load(open(pickle_path))
            return
        #
        self.intData = []
        max_qlen = -1
        for it_data,qa_pair in enumerate(self.data):
            i_qlen = len(qa_pair['question'])
            if(i_qlen > max_qlen):
                max_qlen = i_qlen

        #Pdb().set_trace()
        for it_data,qa_pair in enumerate(self.data):
            q1 = self.prepare_sequence(qa_pair['question'],VQADataSet.qwtoi,max_qlen)
            a1 = VQADataSet.atoi.get(qa_pair['answer'],VQADataSet.atoi['UNK'])
            img_id = qa_pair['image_id']
            self.intData.append((q1,a1,img_id))
        #
        print('pickle int data')
        pickle.dump(self.intData,open(pickle_path,'w'))

    def populateVocabsFromFile(self):
        if VQADataSet.qwtoi is None:
            VQADataSet.qwtoi = pickle.load(open(os.path.join(self.rootDir,'ques_vocab.pkl')))
            VQADataSet.qitow = pickle.load(open(os.path.join(self.rootDir,'ques_rev_vocab.pkl')))
            VQADataSet.atoi = pickle.load(open(os.path.join(self.rootDir,'ans_vocab_'+str(self.max_answers)+'.pkl')))
            VQADataSet.itoa = pickle.load(open(os.path.join(self.rootDir,'ans_rev_vocab_'+str(self.max_answers)+'.pkl')))
            #

    def populateAndWriteVocabs(self,data):
        print("Writing vocabs ..")
        VQADataSet.qwtoi, VQADataSet.qitow = {}, {}
        VQADataSet.atoi, VQADataSet.itoa = {}, {}
        VQADataSet.qwtoi['UNK'], VQADataSet.atoi['UNK'] = len(VQADataSet.qwtoi), len(VQADataSet.atoi)
        VQADataSet.qitow[0], VQADataSet.itoa[0] = 'UNK', 'UNK'
        #
        #answerFreq = {}
        #Pdb().set_trace()
        answerFreq = Counter(map(lambda x: x['answer'], data))
        mostFreq = dict(answerFreq.most_common(self.max_answers))
        for i, datum in enumerate(data):
            for word in datum['question']:
                if word not in VQADataSet.qwtoi:
                    VQADataSet.qwtoi[word] = len(VQADataSet.qwtoi)
                    VQADataSet.qitow[len(VQADataSet.qwtoi)-1] = word

            answer = datum['answer']
            #if answer in answerFreq:
            #    answerFreq[answer]  += 1
            #else:
            #    answerFreq[answer] = 0
            ##
            if answer not in VQADataSet.atoi and answer in mostFreq:
                VQADataSet.atoi[answer] = len(VQADataSet.atoi)
                VQADataSet.itoa[len(VQADataSet.atoi)-1] = answer
        #
        pickle.dump(VQADataSet.qwtoi, open(os.path.join(self.rootDir,'ques_vocab.pkl'),'w'))
        pickle.dump(VQADataSet.atoi, open(os.path.join(self.rootDir,'ans_vocab_'+str(self.max_answers)+'.pkl'),'w'))
        pickle.dump(VQADataSet.qitow,open(os.path.join(self.rootDir,'ques_rev_vocab.pkl'),'w'))
        pickle.dump(VQADataSet.itoa,open(os.path.join(self.rootDir,'ans_rev_vocab_'+str(self.max_answers)+'.pkl'),'w'))


    def prepare_sequence(self,seq, to_ix,max_qlen):
        idxs = [to_ix.get(w,to_ix['UNK']) for w in seq]
        idxs = uc.pad(idxs,max_qlen)
        tensor = torch.LongTensor(idxs)
        return tensor
        #return autograd.Variable(tensor)


    def getQuestionAnswerPairs(self,qfn,afn):
        basedir = os.path.dirname(qfn)
        basename_qfn = os.path.basename(qfn)
        basename_afn = os.path.basename(afn)
        pickle_path = os.path.join(basedir,basename_qfn+'.pkl')
        if os.path.exists(pickle_path):
            #print('unpickle qan.. actually.. chuck it. i dont need self.data ..')
            #return None
            qas = pickle.load(open(pickle_path))
            return qas
        #
        qf = json.load(open(qfn))
        af = json.load(open(afn))

        quess = qf['questions']
        anns = af['annotations']
        qas = []
        for i,ques in enumerate(quess):
            if(self.debug == True and i > 100):
                break
            qa = {}

            qa['question'] = uc.insertSpacesAroundPunctuations(ques['question'].lower()).split()
            qa['question_id'] = ques['question_id']
            qa['image_id'] = ques['image_id']
            qa['answer'] = unicode(uc.removePunctuations(str(anns[i]['multiple_choice_answer'].lower())))
            qas.append(qa)

        print('Pickle qan')
        pickle.dump(qas,open(pickle_path,'w'))
        return qas


    def __len__(self):
        #Tracer()()
        return len(self.intData)

    def __getitem__(self, idx):
        #Pdb().set_trace()
        question, answer, image_id = self.intData[self.ind[idx]]

        if self.features_dir is None:
            #img = Image.open(os.path.join(self.idir,"COCO_train2014_"+("%012d"%image_id)+'.jpg'))
            img = Image.open(self.idir+("%012d"%image_id)+'.jpg')
            img = img.convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
        else:
            imgemb_path = os.path.join(self.features_dir,str(image_id))
            img = torch.load(imgemb_path)
        #
        return question, answer, img, image_id
