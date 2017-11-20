# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
import torch
import torch.nn as nn
import torchvision.models as models
from IPython.core.debugger import Pdb
import utils


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        # Pdb().set_trace()
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I', output_size=1024, mode='train'):
        super(ImageEmbedding, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-2]
        if image_channel_type.lower() == 'normi':
            extactor_fc_layers.append(Normalize(p=2))
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

        # TODO: Get rid of this hack
        self.mode = mode

    def forward(self, image):
        # Pdb().set_trace()
        if self.mode not in ['train', 'val']:
            image = self.extractor(image)
        image_embedding = self.fflayer(image)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=batch_first)
            self.fflayer = nn.Sequential(
                nn.Linear(2 * num_layers * hidden_size, output_size),
                nn.Tanh())

    def forward(self, ques):
        _, hx = self.lstm(ques)
        lstm_embedding = torch.cat([hx[0], hx[1]], dim=2)
        ques_embedding = lstm_embedding[0]
        if self.lstm.num_layers > 1:
            for i in range(1, self.lstm.num_layers):
                ques_embedding = torch.cat(
                    [ques_embedding, lstm_embedding[i]], dim=1)
            ques_embedding = self.fflayer(ques_embedding)
        return ques_embedding


class VQAModel(nn.Module):

    def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I', ques_channel_type='LSTM', mode='train', features_dir=None):
        super(VQAModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        self.word_emb_size = word_emb_size
        self.image_channel = ImageEmbedding(
            image_channel_type, output_size=emb_size)

        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        if ques_channel_type.lower() == 'lstm':
            self.ques_channel = QuesEmbedding(
                input_size=word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        elif ques_channel_type.lower() == 'deeplstm':
            self.ques_channel = QuesEmbedding(
                input_size=word_emb_size, output_size=emb_size, num_layers=2, batch_first=False)
        else:
            msg = 'ques channel type not specified. please choose one of -  lstm or deeplstm'
            print(msg)
            raise Exception(msg)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, 1000),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1000, output_size))

    def forward(self, images, questions, image_ids=None):
        image_embeddings = self.image_channel(images)
        embeds = self.word_embeddings(questions)
        ques_embeddings = self.ques_channel(embeds)
        combined = image_embeddings * ques_embeddings
        output = self.mlp(combined)
        return output
