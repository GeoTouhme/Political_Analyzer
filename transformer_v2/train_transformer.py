import sqlite3
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

DB_PATH = '/home/ubuntu/.openclaw/workspace/political_analyzer/transformer_v2/training_data.sqlite'
MODEL_NAME = 'distilbert-base-multilingual-cased'
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class NewsDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, item):
        title = str(self.titles[item])
        label = int(self.labels[item])
        encoding = self.tokenizer(
            title, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length',
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    print("--- TRANSFORMER TRAINING START (V3) ---")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT title, label FROM training", conn)
    conn.close()
    
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data_loader = DataLoader(NewsDataset(df_train.title.to_numpy(), df_train.label.to_numpy(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}...")
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, len(df_train))
        print(f"Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}")

    save_path = '/home/ubuntu/.openclaw/workspace/political_analyzer/transformer_v2/war_model'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"SUCCESS: Model saved to {save_path}")

if __name__ == "__main__":
    main()
