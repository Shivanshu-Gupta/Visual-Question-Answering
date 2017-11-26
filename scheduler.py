
from torch.optim import lr_scheduler


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
    


