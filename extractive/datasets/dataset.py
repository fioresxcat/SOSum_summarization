import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentences = row['answer_body'].split('. ')  # Split text into sentences
        labels = [1 if i in eval(row['truth']) else 0 for i in range(len(sentences))]  # Create labels based on 'truth'
        
        # Tokenize each sentence
        inputs = self.tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        
        return {
            'input_ids': inputs['input_ids'],  # Tokenized inputs
            'attention_mask': inputs['attention_mask'],  # Attention masks
            'labels': torch.tensor(labels),  # Ground truth labels for each sentence
            'sentences': sentences  # Original sentences for evaluation
        }