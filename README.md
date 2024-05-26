# Aspect-Term Polarity Classification in Sentiment Analysis

To install dependencies, run the following command:

```bash
pip install -r requirements.txt
```


Team members:
- Antoine Poupon
- Olav de Clerk
- Hugo de Rohan Willner


Model description:
- Methodology inspired from "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" paper.
- Bert model fine-tuned on ABSA task.
- Context (Aspect, Category, location) added as auxilliary sentence, so as to fine-tune Bert on a sentence pair classification task


Accuracy on the dev dataset: 0.84