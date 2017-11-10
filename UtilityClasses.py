from __future__ import print_function
import numpy as np
#import ImageNetMiniDataLoader as inmd
import torch
from IPython.core.debugger import Tracer

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os,sys,shutil
from torch.optim import lr_scheduler
import VQADataLoader as vqad
#import POSDataLoader as posd
import string, re

re_for_spaces_around_puct = re.compile('([.,!?()])')

def removePunctuations(sent):
    return sent.translate(None,string.punctuation)

def insertSpacesAroundPunctuations(sent):
    return (re_for_spaces_around_puct.sub(r' \1 ', sent))

def save_image_features(features,image_ids,features_dir):
    bs = features.data.shape[0]
    for image_num in range(bs):
        thisFeature = features.data[image_num]
        thisImageId = image_ids.data[image_num]
        fileName = os.path.join(features_dir,str(thisImageId))
        if(not os.path.exists(fileName)):
            torch.save(thisFeature.cpu(),fileName)



def getVQATrainAndValidationLoader(config):
    crop_param = config['data']['crop_params']
    scale_param = tuple(config['data']['scale_params'])
    #
    transform = transforms.Compose(
            [transforms.Scale(scale_param), 
            transforms.CenterCrop(crop_param),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])
    validationTransforms = transforms.Compose([
            transforms.Scale(scale_param),
            transforms.CenterCrop(crop_param),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ]) 

    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    if(config['mode']=='write_features'):
        fdir  = None
    else:
        fdir = config['data']['features_dir']
    trainset = vqad.VQADataSet(rootDir=dataPath, train=True,
                                      transforms=transform,model_class=config['model_class'],debug = config['debug'],batch_size = config['data']['custom_batch_size'],features_dir=fdir)
    # 
    validset = vqad.VQADataSet(rootDir=dataPath, validation=True, 
                 transforms=validationTransforms,model_class = config['model_class'],debug = config['debug'],features_dir=fdir)
    #
    numTrain = len(trainset)
    indices = list(range(numTrain))
    # 
    #if config['data']['shuffle'] == True:
    #    np.random.seed(config['data']['random_seed'])
    #    np.random.shuffle(indices)
    #
    #trainsampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    trainloader = torch.utils.data.DataLoader(trainset, **config['data']['loader_params'])
    validloader = torch.utils.data.DataLoader(validset,**config['data']['loader_params'])
    return(trainloader,validloader)
    pass



def getVQATestLoader(config):
    transform = transforms.Compose(
            [transforms.Scale((256,256)), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])
   
    dataPath = os.path.join(os.getcwd(),config['data']['path'])

    qfn = config['data']['questions_path']
    afn = config['data']['annotation_path']
    imageDir = os.path.join(dataPath,'train2014')
    testset = vqad.VQADataSet(rootDir=dataPath,questionFileName=qfn, annotationFileName=afn,imageDir=imageDir,model_class=config['model_class'], transforms=transform,debug = config['debug'])
    testloader = torch.utils.data.DataLoader(testset,**config['data']['loader_params'])
    return testloader
"""
def getTrainAndValidationLoaderCifar10(config):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.RandomHorizontalFlip()
            ])

    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True,
                                        download=True, transform=transform)
    
    # load the dataset
    #having it seperate because transormations could be different, for now they are same
    validset = torchvision.datasets.CIFAR10(root=dataPath, train=True, 
                download=True, transform=transform)
    numTrain = len(trainset)
    indices = list(range(numTrain))
    split = int(np.floor(config['data']['validation_ratio']* numTrain))
    # 
    if config['data']['shuffle'] == True:
        np.random.seed(config['data']['random_seed'])
        np.random.shuffle(indices)
    #
    trainIdx, validIdx = indices[split:], indices[:split]

    trainsampler = torch.utils.data.sampler.SubsetRandomSampler(trainIdx)
    validsampler = torch.utils.data.sampler.SubsetRandomSampler(validIdx)

    trainloader = torch.utils.data.DataLoader(trainset,sampler=trainsampler, **config['data']['loader_params'])
    validloader = torch.utils.data.DataLoader(validset,sampler=validsampler,**config['data']['loader_params'])
    return(trainloader,validloader)

def getPOSDataLoaderFromFile(config):
    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    sentencePath = config['data']['sentence_path']
    tagPath = config['data']['tag_path']
    testSet = posd.POSDataSet(dataPath,train=False,validation=False,sentenceFileName=sentencePath,tagFileName=tagPath)
    loader = torch.utils.data.DataLoader(testSet,**config['data']['loader_params'])
    return(loader)

def getPOSDataLoader(config):
    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    trainSet = posd.POSDataSet(dataPath,train=True,validation=False)
    validSet = posd.POSDataSet(dataPath,train=False,validation=True)
    numTrain = len(trainSet)
    #indices = list(range(numTrain))
    # 
    #if config['data']['shuffle'] == True:
    #    np.random.seed(config['data']['random_seed'])
    #    np.random.shuffle(indices)
    #
    #trainsampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    #trainloader = torch.utils.data.DataLoader(trainSet,sampler=trainsampler, **config['data']['loader_params'])
    trainloader = torch.utils.data.DataLoader(trainSet, **config['data']['loader_params'])
    validloader = torch.utils.data.DataLoader(validSet,**config['data']['loader_params'])
    return(trainloader,validloader)
"""

def customVQADataBatcher(loader,batchSize=1):
    batchNumber = -1
    
    runningInputs = []
    runningLabels = []
    runningImages = []
    runningImageIds = []

    runningSize = -1
    for i,(si,sl,im,imid) in enumerate(loader,0):
        ts = si.size()[1]
        if (len(runningInputs) != 0) and ((ts != runningSize) or len(runningInputs) == batchSize):
            inputs = torch.cat(runningInputs,0)
            labels = torch.cat(runningLabels,0)
            images = torch.cat(runningImages,0)
            imageIds = torch.cat(runningImageIds,0)
            batchNumber += 1
            yield(batchNumber,(inputs,labels,images,imageIds))
            runningInputs = []
            runningLabels = []
            runningImages = []
            runningImageIds = []
        #
        runningSize = ts
        runningInputs.append(si)
        runningLabels.append(sl)
        runningImages.append(im)
        runningImageIds.append(imid)

def customSentenceBatcher(loader,batchSize=1):
    batchNumber = -1
    runningInputs = []
    runningLabels = []
    runningSize = -1
    for i,(si,sl) in enumerate(loader,0):
        ts = si.size()[1]
        if (len(runningInputs) != 0) and ((ts != runningSize) or len(runningInputs) == batchSize):
            inputs = torch.cat(runningInputs,0)
            labels = torch.cat(runningLabels,0)
            batchNumber += 1
            yield(batchNumber,(inputs,labels))
            runningInputs = []
            runningLabels = []
        #
        runningSize = ts
        runningInputs.append(si)
        runningLabels.append(sl)

        
"""         
def getTrainAndValidationLoaderImagenetMini(config):
    transform = transforms.Compose(
            [transforms.Scale((256,256)), 
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])
    validationTransforms = transforms.Compose([
            transforms.Scale((256,256)),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ]) 

    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    trainset = inmd.ImagenetMiniDataSet(rootDir=dataPath, train=True,
                                      transform=transform)
    # 
    validset = inmd.ImagenetMiniDataSet(rootDir=dataPath, validation=True, 
                 transform=validationTransforms)
    numTrain = len(trainset)
    indices = list(range(numTrain))
    # 
    if config['data']['shuffle'] == True:
        np.random.seed(config['data']['random_seed'])
        np.random.shuffle(indices)
    #
    trainsampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    trainloader = torch.utils.data.DataLoader(trainset,sampler=trainsampler, **config['data']['loader_params'])
    validloader = torch.utils.data.DataLoader(validset,**config['data']['loader_params'])
    return(trainloader,validloader)

def getTestLoaderImagenetMini(config):
    transform = transforms.Compose([
            transforms.Scale((256,256)),
            #transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ]) 

    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    deepDir = True
    if config['data'].has_key('deep_dir'):
        deepDir = config['data']['deep_dir']
    #
    testset = inmd.ImagenetMiniDataSet(rootDir = dataPath,test = True,transform=transform,deepDir=deepDir)
    testloader = torch.utils.data.DataLoader(testset,**config['data']['loader_params'])
    return testloader

"""
def augmentTestBatchImagenetmini(inputs):
    i5 = inputs[:,:,14:14+227,14:14+227] 
    i1 = inputs[:,:,0:227,0:227]
    i2 = inputs[:,:,29:256,29:256]
    i3 = inputs[:,:,0:227,29:256]
    i4 = inputs[:,:,29:256,0:227]
    return [i5,i1,i2,i3,i4]



def getTestLoaderCifar10(config):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.RandomHorizontalFlip()
            ])

    dataPath = os.path.join(os.getcwd(),config['data']['path'])
    testset = torchvision.datasets.CIFAR10(root=dataPath, train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,**config['data']['loader_params'])
    return testloader


class EWMA():
    def __init__(self,memory):
        self.memory = memory
        self.reset()

    def reset(self):
        self.count = 0
        self.normalizingFactor = 1
        self.memoryToPowerCount = 1
        self.sum = 0
        self.x = 0
        self.avg = 0
        self.y1 = 0


    def update(self,x,n = 1):
        self.count += n
        self.x = x
        memoryToPowerN = self.memory**n
        if (self.memory != 1):
            factor =  (1 - memoryToPowerN)/(1 - self.memory)
            self.memoryToPowerCount = self.memoryToPowerCount*memoryToPowerN
            self.normalizingFactor = (1 - self.memoryToPowerCount)/(1 - self.memory) 
        else:
            self.normalizingFactor = self.count
            factor = n
        
        #
        self.sum = (memoryToPowerN)*self.sum + self.x*factor
        self.avg = self.sum/self.normalizingFactor


class CustomReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    def __init__(self,optimizer,maxPatienceToStopTraining=20, kwargs={}):
        super(CustomReduceLROnPlateau, self).__init__(optimizer,**kwargs)
        self.unconstrainedBadEpochs = self.num_bad_epochs
        self.maxPatienceToStopTraining = maxPatienceToStopTraining
        self._init_getThresholdFn(self.mode,self.threshold,self.threshold_mode)

    def _init_getThresholdFn(self, mode, threshold, threshold_mode):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.getThresholdFn = lambda best: (best * rel_epsilon)
        elif mode == 'min' and threshold_mode == 'abs':
            self.getThresholdFn = lambda best: best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.getThresholdFn = lambda best: best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.getThresholdFn = lambda best: best + threshold



    def step(self, metrics, epoch=None):
        if self.is_better(metrics, self.best):
            self.unconstrainedBadEpochs = 0
        else:
            self.unconstrainedBadEpochs += 1
        #
        super(CustomReduceLROnPlateau,self).step(metrics,epoch)
        

    def shouldStopTraining(self):
        self.currentThreshold = self.getThresholdFn(self.best)
        print("Num_bad_epochs: {0}, unconstrainedBadEpochs: {1}, bestMetric: {2}, currentThreshold: {3}".format(self.num_bad_epochs, self.unconstrainedBadEpochs, self.best,self.currentThreshold)) 
        return(self.unconstrainedBadEpochs > self.maxPatienceToStopTraining)
