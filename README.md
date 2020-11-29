![author](https://img.shields.io/badge/author-mohd--faizy-red)
![made-with-Markdown](https://img.shields.io/badge/Made%20with-markdown-blue)
![Language](https://img.shields.io/github/languages/top/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)
![Platform](https://img.shields.io/badge/platform-Visual%20Studio%20Code-blue)
![Maintained](https://img.shields.io/maintenance/yes/2020)
![Last Commit](https://img.shields.io/github/last-commit/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)
[![GitHub issues](https://img.shields.io/github/issues/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/issues)
[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://opensource.com/resources/what-open-source)
![Stars GitHub](https://img.shields.io/github/stars/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)
[![GitHub license](https://img.shields.io/github/license/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)](https://github.com/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT/blob/master/LICENSE)
![Size](https://img.shields.io/github/repo-size/mohd-faizy/06P_Sentiment-Analysis-With-Deep-Learning-Using-BERT)

# Sentiment Analysis with Deep Learning using BERT

<img src='https://thenewsstrike.com/wp-content/uploads/2020/04/Sentiment-Analysis-1024x457.jpg'>


## __What is BERT?__

__Bidirectional Encoder Representations from Transformers (BERT)__ is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google.BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary, where BERT takes into account the context for each occurrence of a given word.

<img src='https://www.researchgate.net/publication/335238076/figure/fig3/AS:793590422855682@1566218109398/Overall-architecture-of-LCF-design-BERT-shared-layer-is-alternative-to-substitute-for.ppm'>

## What are some variants of BERT?

> BERT has inspired many variants: __RoBERTa, XLNet, MT-DNN, SpanBERT, VisualBERT, K-BERT, HUBERT__ and more. Some variants attempt to compress the model: __TinyBERT, ALERT, DistilBERT__ and more. We describe a few of the variants that outperform BERT in many tasks

> RoBERTa: Showed that the original BERT was undertrained. RoBERTa is trained longer, on more data; with bigger batches and longer sequences; without NSP; and dynamically changes the masking pattern.

> ALBERT: Uses parameter reduction techniques to yield a smaller model. To utilize inter-sentence coherence, ALBERT uses Sentence-Order Prediction (SOP) instead of NSP.
XLNet: Doesn't do masking but uses permutation to capture bidirectional context. It combines the best of denoising autoencoding of BERT and autoregressive language modelling of Transformer-XL.

> MT-DNN: Uses BERT with additional multi-task training on NLU tasks. Cross-task data leads to regularization and more general representations.

<img src='https://devopedia.org/images/article/241/9991.1575378177.jpg'>


## __Dataset__
> We will use the [__SMILE Twitter DATASET__](https://doi.org/10.6084/m9.figshare.3187909.v2)

## __Objective__

:one: To Understand what __Sentiment Analysis__ is, and how to approach the problem from a neural network perspective.

:two: Loading in pretrained BERT with custom output layer.

:three: Train and evaluate finetuned BERT architecture on Sentiment Analysis.


## Finetuning BERT in PyTorch for sentiment analysis.
![BERT](https://miro.medium.com/max/700/0*ViwaI3Vvbnd-CJSQ.png)


- __BERT__ is basically the advancement of the __RNNs__, as its able to Parallelize the Processing and Training. For __Example__ - In sentence we have to process each word sequentially, __BERT__ allow us to do the things in Parellel.
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

:black_circle::zero::one: An introduction to some basic theory behind BERT, and the problem we will be using it to solve

:large_blue_circle::zero::two: Explore dataset distribution and some basic preprocessing

:black_circle::zero::three: Split dataset into training and validation using stratified approach

:large_blue_circle::zero::four: Loading pretrained tokenizer to encode our text data into numerical values (tensors)

:black_circle::zero::five: Load in pretrained BERT with custom final layer

:large_blue_circle::zero::six: Create dataloaders to facilitate batch processing

:black_circle::zero::seven: Choose and optimizer and scheduler to control training of model

:large_blue_circle::zero::eight: Design performance metrics for our problem

:black_circle::zero::nine: Create a training loop to control PyTorch finetuning of BERT using CPU or GPU acceleration

:large_blue_circle::one::zero: Loading finetuned BERT model and evaluate its performance

:black_circle::one::one: Oth-Resources

### Connect with me:


[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][StackExchange AI]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[StackExchange AI]: https://ai.stackexchange.com/users/36737/cypher


---


![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)

