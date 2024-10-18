import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pad_sequence
import ast

# Load the CSV file
df = pd.read_csv('/media/manhdd5/4T/Manhdd/SOSum_summarization/data/train_dataset.csv')  # Replace with the actual file path
df['answer_body'] = df['answer_body'].str.replace('[©¥¢]', '', regex=True)  # Preprocessing to remove special characters
df['truth'] = df['truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])

# Tokenizer and Model from BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Dataset class for loading data into PyTorch
class SentenceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.sentences, self.labels = self._prepare_data()
    
    def _prepare_data(self):
        # Split each answer into sentences, creating a sentence-level dataset
        sentences = []
        labels = []
        for idx, row in self.dataframe.iterrows():
            answer_sentences = row['answer_body'].split('.')
            truth_indices = row['truth'] if row['truth'] else []  # Convert to list if needed
            for i, sentence in enumerate(answer_sentences):
                sentences.append(sentence.strip())
                labels.append(1 if i in truth_indices else 0)
        return sentences, labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return encoding, torch.tensor(label)

# Simple feed-forward network for classification
class SentenceClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=1):
        super(SentenceClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        logits = self.fc(pooled_output)
        return torch.sigmoid(logits)

# Custom collate function to pad inputs to the same length within each batch
def collate_fn(batch):
    input_ids = [item[0]['input_ids'].squeeze(0) for item in batch]
    attention_masks = [item[0]['attention_mask'].squeeze(0) for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to the max length in this batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    labels = torch.stack(labels)
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
    }, labels

# Load data and split into train-test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = SentenceDataset(train_df)
test_dataset = SentenceDataset(test_df)

# DataLoader with custom collate function
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.BCELoss()

# Training loop
for epoch in range(30):  # Train for 3 epochs
    model.train()
    running_loss = 0.0
    for encoding, labels in train_loader:
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)}")

# Function to calculate ROUGE scores
def calculate_rouge_scores(pred_summary, true_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(true_summary, pred_summary)
    return scores

# Function to reconstruct the predicted summary based on sentence indices
def get_predicted_summary(all_preds, sentences):
    return ' '.join([sentence for sentence, pred in zip(sentences, all_preds) if pred == 1])

# Function to get true summary from 'truth' indices, with safeguards for out-of-range indices
def get_true_summary(truth_indices, sentences):
    valid_indices = [i for i in truth_indices if i < len(sentences)]  # Only use valid indices
    if not valid_indices:
        return ''  # Return empty string if no valid indices
    return ' '.join([sentences[i] for i in valid_indices])

# Evaluation loop for ROUGE score
model.eval()
all_preds, all_labels, pred_summaries, true_summaries = [], [], [], []
with torch.no_grad():
    for idx, (encoding, labels) in enumerate(test_loader):
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        preds = (outputs > 0.5).cpu().numpy().flatten()

        # Collect predictions and actual labels for classification metrics
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Get the sentences for the current answer
        sentences = test_dataset.sentences[idx]  # Ensure we get the correct sentences for the current sample
        # Reconstruct predicted and true summaries based on indices
        pred_summary = get_predicted_summary(preds, sentences)
        true_summary = get_true_summary(test_df['truth'].iloc[idx], sentences)

        pred_summaries.append(pred_summary)
        true_summaries.append(true_summary)

# Calculate ROUGE scores for each predicted summary vs true summary
rouge_scores = [calculate_rouge_scores(pred, true) for pred, true in zip(pred_summaries, true_summaries)]

# Calculate average ROUGE-1, ROUGE-2, ROUGE-L
rouge1_avg = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
rouge2_avg = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
rougeL_avg = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

print(f"ROUGE-1 Average F1 Score: {rouge1_avg}")
print(f"ROUGE-2 Average F1 Score: {rouge2_avg}")
print(f"ROUGE-L Average F1 Score: {rougeL_avg}")

# Classification metrics (F1, precision, recall)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
