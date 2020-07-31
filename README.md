# Sentiment Analysis with Deep Learning using BERT

__Bidirectional Encoder Representations from Transformers__ 


Finetuning BERT in PyTorch for sentiment analysis.
![BERT](https://miro.medium.com/max/700/0*ViwaI3Vvbnd-CJSQ.png)


- __BERT__ is basically the advancement of the __RNNs__, as its able to Parallelize the Processing and Training. For Example $\rightarrow$ In sentence we have to process each word sequentially, __BERT__ allow us to do the things in Parellel.
- __BERT__ is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.

![Embeddings](https://mengxinji.github.io/Blog/images/bert/embedding.jpg)

> We will be using the __Hugging Face Transformer library__ that provides a __high-level API__ to state-of-the-art transformer-based models such as __BERT, GPT2, ALBERT, RoBERTa, and many more__. The Hugging Face team also happens to maintain another highly efficient and super fast library for text tokenization called Tokenizers.

## __Specification__
    - Bidirectional: Bert is naturally bi-directional
    - Generalizable: Pre-trained BERT model can be fine-tuned easily for downstream NLp task.
    - High Performace: Fine-tuned BERT models beats state-of-art results for many NLP tasks.
    - Universal: Framework pre-trained on a large corpus of unlabelled text that includes the entire Wikipedia( 2,500 million words!) & Book Corpus (800 million words)



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) 

```bash
pip install torch torchvision
pip install tqdm
pip install transformers

```


### 1: An introduction to some basic theory behind BERT, and the problem we will be using it to solve

### 2: Explore dataset distribution and some basic preprocessing

### 3: Split dataset into training and validation using stratified approach

### 4: Loading pretrained tokenizer to encode our text data into numerical values (tensors)

### 5: Load in pretrained BERT with custom final layer

### 6: Create dataloaders to facilitate batch processing

### 7: Choose and optimizer and scheduler to control training of model

### 8: Design performance metrics for our problem

### 9: Create a training loop to control PyTorch finetuning of BERT using CPU or GPU acceleration

### 10: Loading finetuned BERT model and evaluate its performance

### 11: Oth-Resources
