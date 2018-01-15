# ref: https://github.com/zcyang/imageqa-san/blob/master/src/san_att_lstm_twolayer_theano.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils
from IPython.core.debugger import Pdb


class ImageEmbedding(nn.Module):
    def __init__(self, output_size=1024, mode='train',
                 extract_features=False, features_dir=None):
        super(ImageEmbedding, self).__init__()
        # Pdb().set_trace()
        self.cnn = models.vgg16(pretrained=True).features
        # self.cnn = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())) #FIXME: Only temporary for the first experiment

        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, output_size),
            nn.Tanh())
        #
        self.mode = mode
        self.extract_features = extract_features
        self.features_dir = features_dir

    def forward(self, image, image_ids):
        # Pdb().set_trace()
        if not self.extract_features:
            # N * 224 * 224 -> N * 512 * 14 * 14
            image = self.cnn(image)
            # N * 512 * 14 * 14 -> N * 512 * 196 -> N * 196 * 512
            image = image.view(-1, 512, 196).transpose(1, 2)
            if self.features_dir is not None:
                utils.save_image_features(image, image_ids, self.features_dir)
        # N * 196 * 512 -> N * 196 * 1024
        image_embedding = self.fc(image)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size, batch_first=batch_first)

    def forward(self, ques):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(ques)
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding


class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(self, vocab_size, word_emb_size=500, emb_size=1024, att_ff_size=512, output_size=1000,
                 num_att_layers=1, num_mlp_layers=1, mode='train', extract_img_features=True, features_dir=None):
        super(SANModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        self.image_channel = ImageEmbedding(output_size=emb_size, mode=mode, extract_img_features=extract_img_features,
                                            features_dir=features_dir)

        self.word_emb_size = word_emb_size
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.ques_channel = QuesEmbedding(
            word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)

        self.san = nn.ModuleList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_size, output_size))

    def forward(self, images, questions, image_ids):
        image_embeddings = self.image_channel(images, image_ids)
        embeds = self.word_embeddings(questions)
        # nbatch = embeds.size()[0]
        # nwords = embeds.size()[1]

        # ques_embeddings = self.ques_channel(embeds.view(nwords, nbatch, self.word_emb_size))
        ques_embeddings = self.ques_channel(embeds)
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output
