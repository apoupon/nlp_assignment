# Aspect-Term Polarity Classification in Sentiment Analysis

To install dependencies, run the following command:

```bash
pip install -r requirements.txt
```

To train and test the model, run:
```bash
python src/tester.py
```

This project was carried out as part of the NLP course at CentraleSupélec. The working group was made up of Antoine Poupon, Olav de Clerk and Hugo de Rohan Willner.

The goal of this assignment is to implement a model that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. The model takes as input 3 elements: a sentence, a term occurring in the sentence, and its aspect category. For each input triple, it produces a polarity label: positive, negative or neutral.

<p align="center">
  <img src="https://github.com/apoupon/nlp_assignment/blob/main/method_scheme.png?raw=true" alt="Method scheme"/>
</p>


## Model description:

### Type of classification model

Our methodology is inspired from "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" [1], which achieved state-of-the art results on aspect-based sentiment analysis (ABSA) tasks in 2019. 

We thus used the pre-trained uncased BERT-base model [2], which is an encoder-only architecture leveraging bidirectional contextual embeddings, that has 12 Transformer blocks, a hidden layer size of 768, and 12 self-attention heads, for a total number of 110M parameters.

We fine-tune BERT on the provided training dataset by adding a classification layer whose output dimension is K, which is the number of categories. Finally, the probability of each category P by applying a softmax function. 

### Input and feature representation

We use the NLI-B input methodology from [1], which consists of annexing a pseudo-sentence to the review. For example, for the following instance in the dataset:

> positive FOOD#QUALITY pie 74:77 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

We will give a two-fold input to BERT consisting of the review: Wait staff is blantently unappreciative of your business but its the best pie on the UWS! and the pseudo-sentence added as a auxiliary sentence: pie 74:77 FOOD#QUALITY. This will fine-tune BERT on a sentence pair classification task.

### Results 
Accuracy on the dev dataset: **0.84** after the second epoch

## References

[1] Sun, C., Huang, L., & Qiu, X. (2019). Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence. arXiv preprint arXiv:1903.09588.
[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.