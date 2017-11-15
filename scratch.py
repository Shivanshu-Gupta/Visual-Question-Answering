import numpy as np
from collections import Counter

batch_size = 128
ind = np.arange(0,len(trainloader.dataset.intData))
ind[:10]
np.random.seed(1)
np.random.shuffle(ind)
ind = list(ind)
ind.sort(key = lambda x: len(trainloader.dataset.intData[x][0]))
N = len(ind)
block_ids = {}
random_block_ids = list(range(N))
np.random.shuffle(random_block_ids)
#generate a random number between 0 to N - 1
blockid = random_block_ids[0]
block_ids[ind[0]] = blockid
running_count = 1 
for ind_it in range(1,N):
    if running_count >= batch_size or len(trainloader.dataset.intData[ind[ind_it]][0]) != len(trainloader.dataset.intData[ind[ind_it-1]][0]):
        blockid = random_block_ids[ind_it]
        running_count = 0 
    #   
    block_ids[ind[ind_it]] = blockid
    running_count += 1
#   
#ind  = [x for _,x in sorted(zip(block_ids,ind))]
ind.sort(key=lambda x: block_ids[x])

bsfreq = Counter(block_ids.values())

bsd = dict(bsfreq)

uneven = set()

for b in block_ids.values():
    if bsd[b] != 128:
        uneven.add(b)


bid_to_ind = {}
for k in block_ids.keys():
    v = block_ids[k]
    if v in bid_to_ind:
        bid_to_ind[v].append(k)
    else:
        bid_to_ind[v] = [k]



batchNumber = -1
runningInputs = []
runningLabels = []
runningImages = []
runningImageIds = []
runningSize = -1
badc = 0
for i,j in enumerate(trainloader.dataset.ind):
    si,sl,imid = trainloader.dataset.intData[j]
    si = uc.unpad(si)
    ts = si.size()[0]
    if (len(runningInputs) != 0) and ((ts != runningSize) or len(runningInputs) == batchSize):
        batchNumber += 1
        if len(runningInputs) != 128:
            badc += 1
            print(ts,len(runningInputs),i,j)
        #
        runningInputs = []
        runningLabels = []
        runningImageIds = []
    #
    runningSize = ts
    runningInputs.append([si])
    runningImageIds.append(imid)


ql = {}
for i in trainloader.dataset.intData:
    l = len(i[0])
    if l in ql:
        ql[l] = ql[l] + 1
    else:
        ql[l] = 1



map(lambda x: len(trainloader.dataset.intData[x][0]), ind[2560:2562])
map(lambda x: len(trainloader.dataset.intData[x][0]), trainloader.dataset.ind[2560:2562])




