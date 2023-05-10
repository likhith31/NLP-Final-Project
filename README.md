# NLP-Final-Project
## Introduction:
A Question Answering system is an artificial intelligence system th
Mini Network Architectureat provides clear responses to their
inquiries. It analyses user questions using natural language processing, machine learning, and other
cutting-edge technologies before retrieving data from a database or the internet and presenting the
solution in a way that is understandable to humans. By reducing time and effort needed to retrieve
information, these systems employ variety of applications. Question-and-answer systems have evolved
into a crucial tool for effective communication in the modern world, when we have access to a great
amount of information. 

## Depthwise separable convolution:
The main difference between depthwise separable convolutions and regular convolutions is that
depthwise separable convolutions are quicker since they need fewer multiplication operations. To do this,
the convolution procedure is split into the depthwise convolution and the pointwise convolution
processes.
Contrary to normal CNNs, where convolution is performed to all M channels at once, depth-wise
operation only applies convolution to one channel at a time. The filters/kernels used here will thus be Dk
x Dk x 1. Given that the input data has M channels, M such filters are necessary. The output will be Dp x
Dp x M in size. 

![image](https://github.com/likhith31/NLP-Final-Project/assets/68466905/0270b04c-d841-4ba8-b2f0-fffec7c52b0d)

## Mini Network Architecture

<img width="258" alt="image" src="https://github.com/likhith31/NLP-Final-Project/assets/68466905/6adb8b95-82d8-43c3-9e7e-f8f0d0f1f591">

The QANet architecture uses a combination of convolutional neural networks (CNNs) and self-attention
mechanisms to extract information from the input passage and the question. The CNNs capture local
features of the input, while the self-attention mechanisms allow the model to attend to different parts of
the input at different levels of granularity.
QANet is composed of several layers, including embedding layers, convolutional layers, self-attention
layers, and output layers. The embedding layers transform the input text into a sequence of vectors, which
are then fed into the convolutional layers to extract local features. The self-attention layers are used to
compute the global representation of the input by attending to different parts of the sequence. Finally,
the output layers predict the start and end positions of the answer within the input sequence. 

## Transfer Learning Architecture
![image](https://github.com/likhith31/NLP-Final-Project/assets/68466905/4516461e-e0d0-440b-a786-d1ca4a053a46)

A deep neural network architecture called BERT (Bidirectional Encoder Representations from Transformers) is intended to handle a range of natural language processing (NLP) challenges, including language comprehension, sentiment analysis, and question-answering. Being a bidirectional model, it can comprehend the context of a word depending on both the words that come before and after it in a phrase. This is accomplished via a method known as masked language modeling, where part of the input tokens are arbitrarily hidden, and the model is trained to anticipate the hidden characters based on the input sequence's remaining tokens.
BERT is built on a neural network known as the Transformer architecture, which employs self-attention techniques to record the relationships between various portions of the input sequence. By adding task-specific layers on top of the pre-trained model and then training the entire model on the task-specific dataset, the pre-trained BERT model may be fine-tuned on a particular downstream NLP job. On a number of NLP tasks, including sentiment analysis, named entity identification, and question answering, fine-tuning BERT has demonstrated to achieve state-of-the-art performance.

BERT is built on a neural network known as the Transformer architecture, which employs self-attention techniques to record the relationships between various portions of the input sequence. By adding task-specific layers on top of the pre-trained model and then training the entire model on the task-specific dataset, the pre-trained BERT model may be fine-tuned on a particular downstream NLP job. On a number of NLP tasks, including sentiment analysis, named entity identification, and question answering, fine-tuning BERT has demonstrated to achieve state-of-the-art performance.

## Model Evaluation
<img width="719" alt="image" src="https://github.com/likhith31/NLP-Final-Project/assets/68466905/8077bc02-fb0a-4e20-af1e-c68e8e11b139">
