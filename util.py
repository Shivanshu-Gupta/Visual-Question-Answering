# not used in final implementaiton

# write unique words from a set of files to a new file
def build_vocab(filename, vocabfile):
    vocab = set()
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.rstrip('\n').split(' ')
            vocab |= set(tokens)
    idx = 0
    print(vocabfile)
    with open(vocabfile, 'w') as f:
        for token in vocab:
            f.write(token + '\n')
            idx = idx + 1
