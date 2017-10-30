# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Normalize(nn.Module):
        def __init__(self, p=2):
            super(Normalize, self).__init__()
            self.p = p

        def forward(self, x):
            x /= x.norm(p=self.p, dim=1, keepdim=True)
            return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I'):
        super(ImageEmbedding, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        if image_channel_type == 'norm I':
            extactor_fc_layers.append(Normalize(p=2))
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Tanh())

    def forward(self, image):
        image_embedding = self.extractor(image)
        image_embedding = self.fflayer(image_embedding)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
            self.fflayer = nn.Sequential(
                nn.Linear(2 * num_layers * hidden_size, 1024),
                nn.Tanh())
            # self.linear = nn.Linear(2 * num_layers * hidden_size, 1024)

    def forward(self, ques):
        _, hx = self.lstm(ques)
        lstm_embedding = torch.cat([hx[0], hx[1]], dim=2)
        ques_embedding = lstm_embedding[0]
        if self.lstm.num_layers > 1:
            for i in range(1, self.lstm.num_layers):
                ques_embedding = torch.cat([ques_embedding, lstm_embedding[i]], dim=1)
            # ques_embedding = F.tanh(self.linear(ques_embedding))
            ques_embedding = self.fflayer(ques_embedding)
        return ques_embedding


class VQAModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=300, image_channel_type='I', ques_channel_type='LSTM'):
        super(VQAModel, self).__init__()
        self.image_channel = ImageEmbedding(image_channel_type)

        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        if ques_channel_type == 'LSTM':
            self.ques_channel = QuesEmbedding(embedding_dim, num_layers=1, batch_first=False)
        else:
            self.ques_channel = QuesEmbedding(embedding_dim, num_layers=2, batch_first=False)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.5),
            nn.Tanh())

    def forward(self, images, questions):
        embeds = self.word_embeddings(questions)
        image_embeddings = self.image_channel(images)
        ques_embeddings = self.ques_channel(embeds)
        combined = image_embeddings * ques_embeddings
        output = self.mlp(combined)
        return output
