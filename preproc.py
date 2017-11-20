import json
import pickle
# from itertools import izip
from collections import Counter
from torchtext import data

# train_ques_json = json.load(open('v2_OpenEnded_mscoco_train2014_questions.json'))
# train_ques = [q['question'] for q in train_ques_json['questions']]
# train_imgid = [q['image_id'] for q in train_ques_json['questions']]
# train_ans_json = json.load(open('v2_mscoco_train2014_annotations.json'))
# train_ans = [a['multiple_choice_answer'] for a in train_ans_json['annotations']]

# val_ques_json = json.load(open('v2_OpenEnded_mscoco_val2014_questions.json'))
# val_ques = val_ques_json['questions']
# val_ans_json = json.load(open('v2_mscoco_val2014_annotations.json'))
# val_ans = val_ans_json['annotations']


def proprocess(quesfile, ansfile, outfile, ansid=None):
    ques_json = json.load(open(quesfile))
    ques = [q['question'] for q in ques_json['questions']]
    quesid = [q['question_id'] for q in ques_json['questions']]
    imgid = [q['image_id'] for q in ques_json['questions']]
    ans_json = json.load(open(ansfile))
    ans = [a['multiple_choice_answer'] for a in ans_json['annotations']]
    k = 1000
    if ansid is None:
            c = Counter(ans)
            topk = c.most_common(n=k)
            ansid = dict((a[0], i) for i, a in enumerate(topk))
    ans = [ansid[a] if a in ansid else k for a in ans]
    with open(outfile, 'w') as out:
            for q, qid, i, a in zip(ques, quesid, imgid, ans):
                    out.write('\t'.join([str(qid), q, str(i), str(a)]) + '\n')
    return ansid


ansid = proprocess(quesfile='/scratch/cse/phd/csz178058/vqadata/v2_OpenEnded_mscoco_train2014_questions.json',
                   ansfile='/scratch/cse/phd/csz178058/vqadata/v2_mscoco_train2014_annotations.json',
                   outfile='train.tsv')
proprocess(quesfile='/scratch/cse/phd/csz178058/vqadata/v2_OpenEnded_mscoco_val2014_questions.json',
           ansfile='/scratch/cse/phd/csz178058/vqadata/v2_mscoco_val2014_annotations.json',
           outfile='val.tsv', ansid=ansid)


def create_loaders(path, trainfile, valfile):
    def parse_int(tok, *args):
        return int(tok)
    quesid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
    ques = data.Field(include_lengths=True)
    imgid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
    ans = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
    train_data, val_data = data.TabularDataset.splits(path=path, train=trainfile, validation=valfile,
                                                      fields=[('quesid', quesid), ('ques', ques), ('imgid', imgid), ('ans', ans)],
                                                      format='tsv')
    batch_sizes = (1, 1)
    train_loader, val_loader = data.BucketIterator.splits((train_data, val_data), batch_sizes=batch_sizes, repeat=False, sort_key=lambda x: len(x.ques))
    ques.build_vocab(train_data)
    print('vocabulary size: {}'.format(len(ques.vocab.stoi)))
    return train_loader, val_loader


print('creating loaders...')
train_loader, val_loader = create_loaders('datasets/', 'train.tsv', 'val.tsv')


def dump_datasets(loader, outfile, sorted=False):
    examples = []
    for ex in loader:
        examples.append((
            ex.quesid.data[0],
            ex.ques[0].data.cpu().numpy(),
            ex.ques[1][0],
            ex.imgid.data[0],
            ex.ans.data[0]))
    if not sorted:
        # required only for train_loader. Other loaders give examples in sorted order.
        examples.sort(key=lambda ex: ex[2])
    with open(outfile, 'wb') as trainf:
        pickle.dump(examples, trainf)
    print('dumped to {}'.format(outfile))


print('dumping train dataset...')
dump_datasets(train_loader, outfile='datasets/train_data.pkl')
print('dumping val dataset...')
dump_datasets(val_loader, outfile='datasets/val_data.pkl', sorted=True)
