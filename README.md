# Visual-Question-Answering
This repository contains an AI system for the task of **[Visual Question Answering]**: given an image and a question related to the image in natural language, the systems answer the question in natural language from the image scene. The system can be configured to use one of 3 different underlying models:

1. **VQA**: This is the *baseline model* given in the paper [VQA: Visual Question Answering]. It encodes the image by a CNN and the question by an LSTM and then combines these for VQA task. It uses *pretrained vgg16* to get the image embedding (may be further normalised), and a 1 or 2-layered LSTM for the question embedding.
2. **SAN**: This is an *attention based model* described in the paper [Stacked Attention Networks for Image Question Answering]. It incorporates attention on the input image.
3. **MUTAN**: This is a variant of the VQA model where instead of a simple of pointwise-product, the image and question embedding are combined using a a special *Multimodal Tucker fusion* technique described in the paper [MUTAN: Multimodal Tucker Fusion for Visual Question Answering].

## Usage
First download the datasets from [http://visualqa.org/download.html] - all items under *Balanced Real Images* except *Complementary Pairs List*. 
```sh
python main.py --config <config_file_path>
```
The system takes its arguments from the config file that it takes as input. Sample config files have been provided in [config/].

In order to speed up the training, it's possible to preprocess the images in the dataset and store the image embeddings by setting the *emb_dir* and *preprocess* flag.

[Visual Question Answering]: https://vqa.cloudcv.org/
[VQA: Visual Question Answering]: https://arxiv.org/abs/1505.00468
[Stacked Attention Networks for Image Question Answering]: https://arxiv.org/pdf/1511.02274
[MUTAN: Multimodal Tucker Fusion for Visual Question Answering]: https://arxiv.org/abs/1705.06676
[http://visualqa.org/download.html]: http://visualqa.org/download.html
[config/]: https://github.com/Shivanshu-Gupta/Visual-Question-Answering/config
