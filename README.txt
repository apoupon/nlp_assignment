Team members:

- Antoine Poupon
- Olav de Clerck
- Hugo de Rohan Willner

Model description:

Type of classification model

Our methodology is inspired from "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" [1], which achieved state-of-the art results on aspect-based sentiment analysis (ABSA) tasks in 2019. 

We thus used the pre-trained uncased BERT-base model [2], which is an encoder-only architecture leveraging bidirectional contextual embeddings, that has 12 Transformer blocks, a hidden layer size of 768, and 12 self-attention heads, for a total number of 110M parameters.

We fine-tune BERT on the provided training dataset by adding a classification layer whose output dimension is K, which is the number of categories. Finally, the probability of each category P by applying a softmax function. 

Input and feature representation

We use the NLI-B input methodology from [1], which consists of annexing a pseudo-sentence to the review. For example, for the following instance in the dataset:


positive FOOD#QUALITY pie 74:77 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

We will give a two-fold input to BERT consisting of the review: Wait staff is blantently unappreciative of your business but its the best pie on the UWS! and the pseudo-sentence added as a auxiliary sentence: pie 74:77 FOOD#QUALITY. This will fine-tune BERT on a sentence pair classification task.

References

[1] Sun, C., Huang, L., & Qiu, X. (2019). Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence. arXiv preprint arXiv:1903.09588.
[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Accuracy on the dev dataset:

0.84 after the second epoch

