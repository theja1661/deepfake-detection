import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# Enable optimized GPU performance
torch.backends.cudnn.benchmark = True

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
fake_df = pd.read_csv('data/Cleaned_Fake.csv')
real_df = pd.read_csv('data/Cleaned_True.csv')

# Check if required columns exist
required_columns = {'title'}
if not required_columns.issubset(fake_df.columns) or not required_columns.issubset(real_df.columns):
    raise ValueError(f"Dataset must contain the columns: {required_columns}")

# Assign labels: Fake = 0, Real = 1
fake_df['label'] = 0
real_df['label'] = 1

# Combine datasets, drop missing values, and shuffle
data = pd.concat([fake_df[['title', 'label']], real_df[['title', 'label']]], axis=0).dropna().sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data['title'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train_tfidf, y_train)
xgb_preds = xgb_model.predict(X_test_tfidf)

# XGBoost Evaluation
print("\n=== XGBoost Model Evaluation ===")
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# Plot XGBoost Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")
plt.show()

# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset for BERT
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# Create DataLoaders
train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=16, shuffle=False, pin_memory=True)

# Load BERT Model
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
scaler = amp.GradScaler()
gradient_accumulation_steps = 2

# Train BERT Model
num_epochs = 5
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        with amp.autocast():
            inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
            labels = batch['label'].to(device)
            outputs = bert_model(**inputs, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss/len(train_loader):.4f}')

# BERT Evaluation
bert_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(device)
        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

# Print BERT Accuracy & Metrics
print("\n=== BERT Model Evaluation ===")
bert_accuracy = accuracy_score(all_labels, all_preds)
print(f'BERT Accuracy: {bert_accuracy:.4f}')
print("\nBERT Classification Report:\n", classification_report(all_labels, all_preds))

# Plot BERT Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt="d", cmap="Greens", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("BERT Confusion Matrix")
plt.show()

# Prediction Functions
def predict_news_xgboost(news_text):
    prediction = xgb_model.predict(vectorizer.transform([news_text]))[0]
    return prediction  # 0 = Fake, 1 = Real

def predict_news_bert(news_text):
    encoding = tokenizer(news_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        prediction = torch.argmax(bert_model(**encoding).logits, dim=1).item()
    return prediction  # 0 = Fake, 1 = Real

# Main Execution - Continuous Input Loop
if __name__ == "__main__":
    while True:
        news_input = input("\nEnter a news title (or type 'exit' to quit): ")
        if news_input.strip().lower() in ["exit", "quit"]:
            print("Exiting the program. Goodbye!")
            break
        if news_input.strip():
            print("Final Prediction:", "Real News" if predict_news_xgboost(news_input) == 1 else "Fake News")
        else:
            print("Invalid input. Please enter a valid news title.")

torch.save(bert_model.state_dict(), 'fake_news_bert.pth')
