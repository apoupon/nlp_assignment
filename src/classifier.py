from typing import List

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import pandas as pd

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # TODO: change to 3 labels
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)


    # Helper functions:
    def tokenize_function(self, dataset):
      # TODO: change to two inputs
      return self.tokenizer(dataset["sentence"], dataset["pseudo_sentence"] , padding="max_length", truncation=True)
    
    def compute_metrics(self, eval_pred):
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      return {"accuracy": (predictions == labels).mean()}
    
    def preprocess(self, train_filename, dev_filename):
        '''
            Return processed dataset
        '''

        # load csv files
        df_train = pd.read_csv('data/traindata.csv', delimiter='\t', names=['polarity', 'aspect', 'term', 'term_pos', 'sentence'])
        df_dev = pd.read_csv('data/devdata.csv', delimiter='\t', names=['polarity', 'aspect', 'term', 'term_pos', 'sentence'])
        pd.set_option("display.max_colwidth", None)

        # create a new data frame with 3 features: sentence, pseudo-sentence (<term> <aspect>) and label (polarity)
        for row in df_train.itertuples():
            df_train.at[row.Index, 'pseudo_sentence'] = row.term + ' ' + row.aspect
            df_train = df_train[['sentence', 'pseudo_sentence', 'polarity']]
            if row.polarity == 'negative':
                df_train.at[row.Index, 'polarity'] = 0
            elif row.polarity == 'neutral':
                df_train.at[row.Index, 'polarity'] = 1
            elif row.polarity == 'positive':
                df_train.at[row.Index, 'polarity'] = 2
        for row in df_dev.itertuples():
            df_dev.at[row.Index, 'pseudo_sentence'] = row.term + ' ' + row.aspect
            df_dev = df_dev[['sentence', 'pseudo_sentence', 'polarity']]
            if row.polarity == 'negative':
                df_dev.at[row.Index, 'polarity'] = 0
            elif row.polarity == 'neutral':
                df_dev.at[row.Index, 'polarity'] = 1
            elif row.polarity == 'positive':
                df_dev.at[row.Index, 'polarity'] = 2

        df_train = df_train.rename(columns={'polarity': 'labels'})
        df_dev = df_dev.rename(columns={'polarity': 'labels'})

        # convert the data frame to a Dataset
        train_dataset = Dataset.from_pandas(df_train)
        dev_dataset = Dataset.from_pandas(df_dev)

        # combine the train and dev datasets
        sft_dataset = DatasetDict({'train': train_dataset, 'val': dev_dataset})

        return sft_dataset
    
    def preprocess_predict(self, predict_filename):
        '''
            Return processed dataset
        '''

        # load csv files
        # TODO predict filename
        df_predict = pd.read_csv('data/devdata.csv', delimiter='\t', names=['polarity', 'aspect', 'term', 'term_pos', 'sentence'])
        pd.set_option("display.max_colwidth", None)

        # create a new data frame with 3 features: sentence, pseudo-sentence (<term> <aspect>) and label (polarity)
        for row in df_predict.itertuples():
            df_predict.at[row.Index, 'pseudo_sentence'] = row.term + ' ' + row.aspect
            df_predict = df_predict[['sentence', 'pseudo_sentence', 'polarity']]
            if row.polarity == 'negative':
                df_predict.at[row.Index, 'polarity'] = 0
            elif row.polarity == 'neutral':
                df_predict.at[row.Index, 'polarity'] = 1
            elif row.polarity == 'positive':
                df_predict.at[row.Index, 'polarity'] = 2

        df_predict = df_predict.rename(columns={'polarity': 'labels'})

        # convert the data frame to a Dataset
        predict_dataset = Dataset.from_pandas(df_predict)

        return predict_dataset
  
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Load the dataset
        # TODO link to path
        dataset = self.preprocess('name','name')
        
        # Tokenise the dataset
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        # Process the dataset
        # TODO: Adapt to our dataset
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets = tokenized_datasets.remove_columns(["pseudo_sentence"])
        tokenized_datasets.set_format("torch")

        # Seperate the dataset
        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        small_eval_dataset = tokenized_datasets["val"].shuffle(seed=42)


        # Create the dataloader
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=2)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

        # Get the model and move to device
        model = self.model
        model.to(device)
        print('training on: ', device)


        # Training setup
        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Let's train this BadBoy
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            loss_list = []
            for idx, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                loss_list.append(loss.item())

                if idx % 50 == 0:
                    print(f"Epoch {epoch} loss: {np.mean(loss_list)}")
                    loss_list = []
        
            # Evaluate the model
            model.eval()
            accuracies = []
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits.cpu()
                    labels = batch["labels"].cpu()
                    predictions = np.argmax(logits, axis=-1)
                    if predictions == labels:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)
            
            # Print the avg accuracy for this epoch
            print(f"Epoch {epoch} mean accuracy: {np.mean(accuracies)}")






    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        # Preprocess
        dataset = self.preprocess_predict('name')




