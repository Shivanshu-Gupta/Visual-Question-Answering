
from __future__ import print_function
from datetime import datetime as dt

import yaml,torch,shutil
from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


import os,sys
import UtilityClasses as uc

import time
import argparse
parser = argparse.ArgumentParser()
from IPython.core.debugger import Pdb

import vqa
import san

parser.add_argument('-c','--config',type=str,default='config.yml')

def prefixId(config):
    prefix = config['prefix']
    if not config.has_key('model_name'):
        config['model_name'] = prefix
    #
    save_dir = os.path.join(os.getcwd(),config['save_dir'])
    save_dir = os.path.join(save_dir,prefix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for mk in ['checkpoints','stats']:
        if config.has_key(mk):
            for k in config[mk].keys():
                f = config[mk][k]
                config[mk][k] = os.path.join(save_dir,f)
    #
    return config

def printHeaders(config):
    #print("time,epoch,nbatches,nsentences,nwords,top1,top5,loss,trainingTime")
    if(config['stats'].has_key('trainTimeLogs')):
        fh = open(config['stats']['trainTimeLogs'],'a+')
        print("time,epoch,nbatches,nsentences,nwords,top1,top5,loss,trainingTime,dataLoadingTime,lr,whichModel",file =fh)
        fh.close()
    #
    fh = open(config['stats']['accuracyLogs'],'a+')
    print("time,epoch,nbatches,nsentences,nwords,top1,top5,loss,dataLoadingTime,trainingTime,lr,whichSet,whichModel",file=fh)
    print("time,epoch,nbatches,nsentences,nwords,top1,top5,loss,dataLoadingTime,trainingTime,lr,whichSet,whichModel")

def accuracy(answer_scores,labels,topk = (1,),dataloader=None,writeToFile=False,fh=None):
    maxk = max(topk)
    batchSize = labels.size()[0]
    #nwords = labels.size()[1]
    #Pdb().set_trace()
    _, pred = answer_scores.topk(maxk)
    pred = pred.view(-1,maxk).t()
    correct = pred.eq(labels.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / (batchSize)))
    #
    if dataloader is not None and batchSize == 1:
        print(' '.join([dataloader.dataset.itoa[x] for x in pred[0,:]]),file=fh)
    #
    return res


def save_image_features(config,dataloader,net):
    net.eval()
    asyncVar = config['data']['loader_params'].has_key('pin_memory') and config['data']['loader_params']['pin_memory']
    #for i, (inputs, labels, images,image_ids) in enumerate(dataloader, 0):
    for i,(inputs,labels,images,image_ids) in uc.customVQADataBatcher(dataloader,config['data']['custom_batch_size']):
        #if i == 0:
        #    break
        dtstr = dt.now()
        if i%10 == 0:
            print(dtstr,i)

        #print('batch size: ',inputs.size()[0], ' nwords: ',inputs.size()[1])
        inputs, labels,images = Variable(inputs,volatile=True), Variable(labels,volatile=True), Variable(images,volatile=True)
        image_ids = Variable(image_ids,volatile=True)
        features_exist = True
        num_missing = 0
        for id in image_ids:
            if not os.path.exists(os.path.join(config['data']['features_dir'],str(id.data[0]))):
                features_exist = False
                num_missing += 1
        if not features_exist:
            print('how many missing: ',num_missing)
            if torch.cuda.is_available():
                inputs,labels = inputs.cuda(), labels.cuda(async=asyncVar)
                images = images.cuda()
                image_ids=  image_ids.cuda()
        
            outputs = net(images,inputs,image_ids)
            


def validate(config,dataloader,net,criterion,optimizer,epoch,whichSet,whichModel):
    batchTime = uc.EWMA(1)
    losses = uc.EWMA(1)
    top1 = uc.EWMA(1)
    top5 = uc.EWMA(1)
    dataLoadingTime = uc.EWMA(1)
    net.eval()
    asyncVar = config['data']['loader_params'].has_key('pin_memory') and config['data']['loader_params']['pin_memory']
    end = time.time()
    nsentences= 0
    writeToFile = False
    if whichSet == 'validationSet':
        fileName= config['stats']['outputOverValidation']
        writeToFile = config['training']['write_output_to_file']
        ofh= open(fileName,'w+')

    customBatchSize = 1
    if(whichSet.lower() == 'trainset'):
        customBatchSize = config['data']['custom_batch_size']

    print('batch size: {0}'.format(customBatchSize))
    for i,(inputs,labels) in uc.customSentenceBatcher(dataloader,customBatchSize):
    #for i, (inputs, labels) in enumerate(dataloader, 0):
        dataLoadingTime.update(time.time() - end) 
         
        if torch.cuda.is_available():
            inputs,labels = inputs.cuda(), labels.cuda(async=asyncVar)

        #
        batchSize = labels.size()[0]
        nwords = labels.size()[1]
        nsentences  += batchSize
        labels = torch.autograd.Variable(labels, volatile=True)
        #need to extract 5 datasets from current batch and take average for prediction
        inputs = torch.autograd.Variable(inputs, volatile=True)
        # compute output
        output = net(inputs)
        #
        loss = criterion(output.view(-1,config['data']['tagset_size']), labels.view(-1))
        if writeToFile:
            prec1, prec5 = accuracy(output.data, labels.data, topk=(1, 5),dataloader=dataloader,writeToFile=writeToFile,fh = ofh)
        else:
            prec1, prec5 = accuracy(output.data, labels.data, topk=(1, 5))

        losses.update(loss.data[0],batchSize*nwords )
        top1.update(prec1[0], batchSize*nwords)
        top5.update(prec5[0], batchSize*nwords)
        # measure elapsed time
        batchTime.update(time.time() - end)
        end = time.time()
    #
    lr = getLearningRate(optimizer)
    fh = open(config['stats']['accuracyLogs'],'a+')
    dtstr = str(dt.now())
    #print("time,epoch,nbatches,nsentences,nwords,top1,top5,loss,trainingTime")
    print("{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8:.2f},{9:.2f},{10:.5f},{11},{12}".format(dtstr,epoch, batchTime.count,nsentences,top1.count, top1.avg, top5.avg, losses.avg, dataLoadingTime.sum,batchTime.sum,lr,whichSet,whichModel ),file=fh)
    print("{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8:.2f},{9:.2f},{10:.5f},{11},{12}".format(dtstr,epoch, batchTime.count,nsentences,top1.count, top1.avg, top5.avg, losses.avg, dataLoadingTime.sum,batchTime.sum,lr,whichSet,whichModel ))
    fh.close()
    return (losses.avg,top1.avg)


def getLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


    

def train(config,trainloader,net,criterion,optimizer,epoch):
    net.train()
    #Pdb().set_trace()
    asyncVar = config['data']['loader_params'].has_key('pin_memory') and config['data']['loader_params']['pin_memory']
    dataLoadingTime = uc.EWMA(1)
    trainingTime = uc.EWMA(1)
    losses = uc.EWMA(1)
    top1 = uc.EWMA(1)
    top5 = uc.EWMA(1)
    end = time.time()
    #Tracer()()
    nsentences= 0
    for i, (inputs, labels, images, image_ids) in enumerate(trainloader, 0):
    #inputs refers to questions, labels refers to gt answers

    #for i,(inputs,labels) in uc.customSentenceBatcher(trainloader,config['data']['custom_batch_size']):
        # get the inputs
        dataLoadingTime.update(time.time() - end) 
        batchSize = labels.size()[0]
        #nwords = labels.size()[1]
        nsentences += batchSize
        #print("batchNumber: {0}, inputSize: {1}, labelSize:{2}, nsentences: {3}".format(i,str(inputs.size()), str(labels.size()), nsentences))
        # wrap them in Variable
        inputs, labels,images = Variable(inputs), Variable(labels), Variable(images)
        if torch.cuda.is_available():
            inputs,labels = inputs.cuda(), labels.cuda(async=asyncVar)
            images = images.cuda()

        outputs = net(images,inputs)
        #loss = criterion(outputs.view(-1,config['data']['tagset_size']), labels.view(-1))
        loss = criterion(outputs, labels)
        # measure accuracy and record loss
        evaluationStartTime = time.time()
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        losses.update(loss.data[0], batchSize)
        top1.update(prec1[0], batchSize)
        top5.update(prec5[0], batchSize)
        
        #compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainingTime.update(time.time() - end)
        end = time.time()
        dtstr = str(dt.now())
        if config.has_key('verbose') and config['verbose'] and i%100 == 0:
            print("{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8},{9}".format(dtstr,epoch, trainingTime.count,nsentences,top1.count, top1.avg, top5.avg, losses.avg, trainingTime.sum,config['model_name'] ))
        
        #if(i == 10):
        #    break
    #
    #write to file and flush
    fh = open(config['stats']['trainTimeLogs'],'a+')

    lr = getLearningRate(optimizer)
    dtstr = str(dt.now())
    print("Exited")

    print("{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8:.2f},{9:.2f},{10:.5f},{11}".format(dtstr,epoch, trainingTime.count,nsentences,top1.count, top1.avg, top5.avg, losses.avg, trainingTime.sum,dataLoadingTime.sum,lr,config['model_name'] ))
    fh = open(config['stats']['trainTimeLogs'],'a+')
    print("{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8:.2f},{9:.2f},{10:.5f},{11}".format(dtstr,epoch, trainingTime.count,nsentences,top1.count, top1.avg, top5.avg, losses.avg, trainingTime.sum,dataLoadingTime.sum,lr,config['model_name'] ),file=fh)
    fh.close()



def saveCheckpoint(state, isBest, config):
    torch.save(state, config['checkpoints']['path'])
    if isBest:
        print("isBest True. Epoch: {0}, bestError: {1}".format(state['epoch'],state['bestError']))
        shutil.copyfile(config['checkpoints']['path'],config['checkpoints']['best_model_path'])


def getLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




def jobManager():
    #3 modes - 
        # train 1 model whose configuration is specified in the given config file 
        # test 1 model whose configuration is specified in the given config file
        # test bulk models - test all models specified in the given config file. All models are specified with their respective config files and an optional path to best model
    global args
    args = parser.parse_args()
    args.config = os.path.join(os.getcwd(),args.config)
    config = yaml.load(open(args.config))
    if not config.has_key('mode'):
        config['mode'] = 'train'

    #Pdb().set_trace()
    config =  prefixId(config)
    printHeaders(config)
    if config['mode'].lower() == 'write_features':
        main(config,'write_features')
    if config['mode'].lower() == 'train':
        main(config,'train')
    elif config['mode'].lower() == 'test':
        main(config,'test',onlyTest=True)
    elif config['mode'].lower() == 'bulk_test':
        bulkTest(config)
    else:
        print("Incorrect mode. Expected train, test or bulk_test")
        
    print("Job Manager Finished")
 
def bulkTest(config):
    outputFilePath = config['stats']['accuracyLogs']
    for model in config['testing']['models']:
        thisConfigPath = model['config_path']
        thisModelPath = model['path']
        thisModelName = model['model_name']
        
        thisConfig  = yaml.load(open(thisConfigPath))
        thisConfig['stats']['accuracyLogs'] = outputFilePath
        thisConfig['checkpoints']['best_model_path'] = thisModelPath
        #if(not thisConfig.has_key('model_name')):
        thisConfig['data']['augment_test'] = config['data']['augment_test']
        thisConfig['model_name'] = thisModelName
        main(thisConfig,'test')


def enhance_config(config):
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    use_gpu = config['use_gpu']
    config['model']['model_class'] = config['model_class']
    config['data']['model_class'] = config['model_class']
    config['data']['features_dir'] = os.path.join(config['data']['path'],config['data']['features_dir'])
    config['model']['params']['features_dir'] = config['data']['features_dir']
    config['model']['mode'] = config['mode']

    if not os.path.exists(config['data']['features_dir']):
        os.makedirs(config['data']['features_dir'])

"""
Test script
from __future__ import print_function
from datetime import datetime as dt

import yaml,torch,shutil
from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


import os,sys
import UtilityClasses as uc

import time
import argparse
parser = argparse.ArgumentParser()
from IPython.core.debugger import Pdb

import vqa
import san

import main1 as m
config = yaml.load(open('config/config3.yml'))
m.enhance_config(config)


"""



def main(config,mode,onlyTest = False):
    use_gpu = config['use_gpu']
    enhance_config(config)

    #Pdb().set_trace()
    if not onlyTest:
        (trainloader,validationloader) = uc.getVQATrainAndValidationLoader(config)
        #testloader = uc.getVQATestLoader(config)
        vocab_size = trainloader.dataset.q_vocab_size
        output_size = trainloader.dataset.a_vocab_size
         
    else:
        testloader = uc.getPOSDataLoaderFromFile(config)
        vocab_size = testloader.dataset.q_vocab_size
        output_size = testloader.dataset.a_vocab_size

    config['model']['params']['vocab_size'] = vocab_size
    config['model']['params']['output_size'] = output_size

    if config['model']['model_class'] == 'vqa':
        net = vqa.VQAModel(**config['model']['params'])
    elif config['model']['model_class'] == 'san':
        net = san.SANModel(**config['model']['params'])
    
    criterion = nn.NLLLoss()

    if(config['optim']['class'].lower() == 'sgd'):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),**config['optim']['params'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),**config['optim']['params'])
    
    if(use_gpu):
        print("Cuda is True.....converting model and loss fn to .cuda")
        net = net.cuda()
        criterion = criterion.cuda()

    startEpoch = 0
    bestPrec1 = 0
    bestError = 1
    
    pathForTrainedModel = config['checkpoints']['path']
    
    if (mode == 'test'):
        pathForTrainedModel = config['checkpoints']['best_model_path']
    
    if ((config['training']['start_from_checkpoint']) or (mode == 'test')):
        if os.path.exists(pathForTrainedModel):
            print("=> loading checkpoint/model found at '{0}'".format(pathForTrainedModel))
            checkpoint = torch.load(pathForTrainedModel)
            startEpoch = checkpoint['epoch']
            bestError = checkpoint['bestError']
            net.load_state_dict(checkpoint['stateDict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{0}'".format(pathForTrainedModel))
            if mode == 'test':
                print(" = > Error: could not find any model at the given path. Quitting...")
                return
   
    if mode == 'write_features':
        save_image_features(config,trainloader,net)


    if mode == 'test':
        epoch = startEpoch - 1
        testLoss,testPrec1 = validate(config,testloader, net, criterion, optimizer,epoch,'testSet',config['model_name'])  
        if onlyTest:
            return
        validLoss,validPrec1 = validate(config,validationloader, net, criterion,optimizer,epoch,'validationSet',config['model_name'])
        trainLoss,trainPrec1 = validate(config,trainloader, net, criterion,optimizer,epoch,'trainSet',config['model_name'])
        return

    #
    #Tracer()
    #Pdb().set_trace()
    scheduler = uc.CustomReduceLROnPlateau(optimizer,config['optim']['scheduler_params']['maxPatienceToStopTraining'], config['optim']['scheduler_params']['base_class_params'])

    print("Start from epoch#:",startEpoch)
    
    
    for epoch in range(startEpoch, config['training']['no_of_epochs']):
        #adjustLearningRate(config,optimizer,epoch)
        train(config, trainloader, net, criterion, optimizer, epoch)
         #evaluate on train set and validation set
        validLoss1,validPrec1 = validate(config,validationloader, net, criterion,optimizer,epoch,'validationSet',config['model_name'])
        trainLoss1,trainPrec1 = validate(config,trainloader, net, criterion,optimizer,epoch,'trainSet',config['model_name'])
        # remember best prec@1 and save checkpoint
        validError = 1.0 - validPrec1
        isBest = validError < bestError
        bestError = min(validError, bestError)
        print("Input to scheduler step: {0}".format(validError))
        scheduler.step(validError,epoch=epoch)
        saveCheckpoint({
            'epoch': epoch + 1,
            'stateDict': net.state_dict(),
            'bestError': bestError,
            'optimizer' : optimizer.state_dict()
            #'scheduler': scheduler.state_dict()
        }, isBest,config)

        if(scheduler.shouldStopTraining()):
            print("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(scheduler.unconstrainedBadEpochs,scheduler.maxPatienceToStopTraining))
            #Pdb().set_trace()
            break
    #
    print('Finished Training')

if __name__ == '__main__':
    #Pdb().set_trace()
    jobManager()
