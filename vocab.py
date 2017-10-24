# reference: vocab object from harvardnlp/opennmt-py

class Vocab(object):
    def __init__(self, filename=None, lower=False, unkWord='unk'):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower

        if filename is not None:
            self.loadFile(filename)

        # have only 'unk' as special word
        idx = self.add(unkWord)
        self.unk = idx

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            token = line.rstrip('\n')
            self.add(token)

    def getIndex(self, key, default=None):
        if self.lower:
            key = key.lower()
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label):
        if self.lower:
            label = label.lower()

        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel) + 1
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def toIdx(self, labels):
        vec = [self.getIndex(label, default=self.unk) for label in labels]
        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def toLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels
