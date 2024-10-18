import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from rouge_score import rouge_scorer
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast

# Load CSV file
df = pd.read_csv('/media/manhdd5/4T/Manhdd/SOSum_summarization/data/train_dataset.csv')  # Replace with the actual file path
df['answer_body'] = df['answer_body'].str.replace('[©¥¢]', '', regex=True)  # Preprocessing to remove special characters
df['truth'] = df['truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])

val_df = pd.read_csv('/media/manhdd5/4T/Manhdd/SOSum_summarization/data/validation_dataset.csv')  # Replace with the actual file path
val_df['answer_body'] = val_df['answer_body'].str.replace('[©¥¢]', '', regex=True)  # Preprocessing to remove special characters
val_df['truth'] = val_df['truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode sentences using BERT
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze(0).numpy()

# Preprocessing step: Encode all sentences into embeddings
def preprocess_data(dataframe):
    sentences = []
    labels = []
    embeddings = []
    
    for idx, row in dataframe.iterrows():
        answer_sentences = row['answer_body'].split('.')
        truth_indices = row['truth']
        for i, sentence in enumerate(answer_sentences):
            if sentence.strip():  # Skip empty sentences
                sentences.append(sentence.strip())
                labels.append(1 if i in truth_indices else 0)  # Label sentences
                embeddings.append(encode_sentence(sentence.strip()))
                
    return np.array(embeddings), np.array(labels)

# # Preprocess the data
# X, y = preprocess_data(df)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = preprocess_data(df)
X_test, y_test = preprocess_data(val_df)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Classifier: {name}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Function to calculate ROUGE scores
def calculate_rouge_scores(pred_summary, true_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(true_summary, pred_summary)
    return scores

# Function to reconstruct the predicted summary based on sentence indices
def get_predicted_summary(sentences, y_pred):
    return ' '.join([sentence for sentence, pred in zip(sentences, y_pred) if pred == 1])

# Function to get true summary based on ground truth
def get_true_summary(sentences, truth_indices):
    return ' '.join([sentences[i] for i in truth_indices])

# Evaluate ROUGE scores for the test set
for idx, row in df.iterrows():
    test_sentences = row['answer_body'].split('.')
    truth_indices = row['truth']
    
    # Use one of the classifiers (e.g., Logistic Regression)
    y_pred = classifiers['MLP Classifier'].predict(scaler.transform([encode_sentence(s) for s in test_sentences if s.strip()]))

    pred_summary = get_predicted_summary(test_sentences, y_pred)
    true_summary = get_true_summary(test_sentences, truth_indices)
    
    rouge_scores = calculate_rouge_scores(pred_summary, true_summary)
    print(f"ROUGE Scores for Answer ID {row['answer_id']}:")
    print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure}")
    print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure}")
    print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure}")
    print("-" * 50)
