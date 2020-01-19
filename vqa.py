# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
import torch
import torch.nn as nn
import torchvision.models as models
import utils
import torch.nn.functional as F

from IPython.core.debugger import Pdb


class MutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        # Pdb().set_trace()
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I', output_size=1024, mode='train',
                 extract_features=False, features_dir=None):
        super(ImageEmbedding, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        if image_channel_type.lower() == 'normi':
            extactor_fc_layers.append(Normalize(p=2))
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

        # TODO: Get rid of this hack
        self.mode = mode
        self.extract_features = extract_features
        self.features_dir = features_dir

    def forward(self, image, image_ids):
        # Pdb().set_trace()
        if not self.extract_features:
            image = self.extractor(image)
            if self.features_dir is not None:
                utils.save_image_features(image, image_ids, self.features_dir)

        image_embedding = self.fflayer(image)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        # TODO: take as parameter
        self.bidirectional = True
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                batch_first=batch_first, bidirectional=self.bidirectional)

            if self.bidirectional:
                self.fflayer = nn.Sequential(
                    nn.Linear(2 * num_layers * hidden_size, output_size),
                    nn.Tanh())
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
        if self.lstm.num_layers > 1 or self.bidirectional:
            for i in range(1, self.lstm.num_layers):
                ques_embedding = torch.cat(
                    [ques_embedding, lstm_embedding[i]], dim=1)
            ques_embedding = self.fflayer(ques_embedding)
        return ques_embedding


class VQAModel(nn.Module):

    def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I', ques_channel_type='lstm', use_mutan=True, mode='train', extract_img_features=True, features_dir=None):
        super(VQAModel, self).__init__()
        self.mode = mode
        self.word_emb_size = word_emb_size
        self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size, mode=mode,
                                            extract_features=extract_img_features, features_dir=features_dir)

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
        if use_mutan:
            self.mutan = MutanFusion(emb_size, emb_size, 5)
            self.mlp = nn.Sequential(nn.Linear(emb_size, output_size))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(emb_size, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, output_size))

    def forward(self, images, questions, image_ids):
        image_embeddings = self.image_channel(images, image_ids)
        embeds = self.word_embeddings(questions)
        ques_embeddings = self.ques_channel(embeds)
        if hasattr(self, 'mutan'):
            combined = self.mutan(ques_embeddings, image_embeddings)
        else:
            combined = image_embeddings * ques_embeddings
        output = self.mlp(combined)
        return output
