# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
import torch,os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from IPython.core.debugger import Pdb
import UtilityClasses as uc

class Normalize(nn.Module):
        def __init__(self, p=2):
            super(Normalize, self).__init__()
            self.p = p

        def forward(self, x):
            x /= x.norm(p=self.p, dim=1, keepdim=True)
            return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I', output_size=1024):
        super(ImageEmbedding, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        if image_channel_type.lower() == 'normi':
            extactor_fc_layers.append(Normalize(p=2))
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        #if torch.cuda.is_available():
        #    self.extractor = self.extractor.cuda()
        #
        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

    def forward(self, image):
        #Pdb().set_trace()
        image_embedding = self.extractor(image)
        image_embedding = self.fflayer(image_embedding)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
            self.fflayer = nn.Sequential(
                nn.Linear(2 * num_layers * hidden_size, output_size),
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

    def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I', ques_channel_type='LSTM',mode='train',features_dir=None):
        super(VQAModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        self.word_emb_size = word_emb_size
        self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size)

        """
        if self.features_dir is not None:
            dir_name  = os.path.dirname(self.features_dir)
            base_name = os.path.dirname(self.features_dir)
            base_name = base_name + '_'+ ''.join(image_channel_type.split())  + '_' + str(emb_size)
            self.features_dir = os.path.join(dir_name,base_name)
        """
        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        #self.word_embeddings = nn.Embedding(vocab_size, word_emb_size, padding_idx=1)
        if ques_channel_type == 'LSTM':
            self.ques_channel = QuesEmbedding(input_size=word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        else:
            self.ques_channel = QuesEmbedding(input_size=word_emb_size, output_size=emb_size, num_layers=2, batch_first=False)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, 1000),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1000, output_size),
            nn.Dropout(p=0.5),
            nn.Tanh())

    """
    def save_image_features(self,features,image_ids,features_dir):
        bs = features.data.shape[0]
        for image_num in range(bs):
            thisFeature = features.data[image_num]
            thisImageId = image_ids.data[image_num]
            fileName = os.path.join(features_dir,str(thisImageId))
            if(not os.path.exists(fileName)):
                torch.save(thisFeature.cpu(),fileName)
    """

    def forward(self, images, questions, image_ids = None):
        #Pdb().set_trace()
        if self.mode == 'write_features':
            image_embeddings = self.image_channel(images)
            uc.save_image_features(image_embeddings,image_ids,self.features_dir)
            return 0
        if self.mode == 'test':
            image_embeddings = self.image_channel(images)
        else:
            image_embeddings = images
        #
        embeds = self.word_embeddings(questions)
        nbatch = embeds.size()[0]
        nwords = embeds.size()[1]
        ques_embeddings = self.ques_channel(embeds.view(nwords,nbatch,self.word_emb_size))
        #ques_embeddings = self.ques_channel(embeds)
        combined = image_embeddings * ques_embeddings
        output = self.mlp(combined)
        return output
