"""
Fine-tuning RuBERT (DeepPavlov/rubert-base-cased) для классификации депрессивных текстов.

Архитектура:
1. RuBERT как feature extractor + classification head
2. Или полный fine-tuning с заморозкой первых N слоёв
3. Сравнение с baseline (Logistic Regression на embeddings)

Требования:
    pip install transformers torch scikit-learn pandas numpy
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from dataset_loader import create_demo_dataset, create_train_test_split

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================

MODEL_NAME = "DeepPavlov/rubert-base-cased"  # RuBERT от DeepPavlov
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Устройство: {DEVICE}")
print(f"Модель: {MODEL_NAME}")

# ==========================================
# DATASET
# ==========================================

class TextDataset(Dataset):
    """PyTorch Dataset для текстов."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }


# ==========================================
# ВАРИАНТ 1: Извлечение эмбеддингов + LogReg (baseline)
# ==========================================

def extract_rubert_embeddings(texts, model, tokenizer, batch_size=16, max_length=256):
    """Извлекает [CLS]-эмбеддинги из RuBERT."""
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
    
    return np.array(embeddings)


def train_baseline_logreg(X_train, y_train, X_test, y_test):
    """Baseline: Logistic Regression на RuBERT-эмбеддингах."""
    print("\n" + "="*60)
    print("BASELINE: Logistic Regression на RuBERT [CLS]")
    print("="*60)
    
    print("Загружаем RuBERT (только энкодер)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    print("Извлекаем эмбеддинги train...")
    X_train_emb = extract_rubert_embeddings(X_train, model, tokenizer)
    print("Извлекаем эмбеддинги test...")
    X_test_emb = extract_rubert_embeddings(X_test, model, tokenizer)
    
    print(f"Размерность эмбеддингов: {X_train_emb.shape}")
    
    print("Обучаем Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_emb, y_train)
    
    y_pred = clf.predict(X_test_emb)
    y_pred_proba = clf.predict_proba(X_test_emb)[:, 1]
    
    print(f"\nAccuracy: {(y_pred == y_test).mean():.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Normal', 'Suicide'])}")
    
    return clf, X_train_emb, X_test_emb, y_pred_proba


# ==========================================
# ВАРИАНТ 2: Полный fine-tuning RuBERT
# ==========================================

class RuBERTClassifier(nn.Module):
    """RuBERT + classification head."""
    
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)


def train_rubert_finetune(X_train, y_train, X_test, y_test):
    """Fine-tuning RuBERT с полным обучением."""
    print("\n" + "="*60)
    print("FINE-TUNING: RuBERT с классификационной головой")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Датасеты
    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Модель
    model = RuBERTClassifier(MODEL_NAME).to(DEVICE)
    
    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_f1 = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].cpu().numpy()
                
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        f1 = f1_score(all_labels, all_preds)
        print(f"  Test F1: {f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
    
    # Load best
    if best_state:
        model.load_state_dict(best_state)
    
    # Final eval
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {(all_preds == np.array(all_labels)).mean():.3f}")
    print(f"F1-score: {f1_score(all_labels, all_preds):.3f}")
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.3f}")
    print(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=['Normal', 'Suicide'])}")
    
    # Save model
    os.makedirs("rubert_model", exist_ok=True)
    torch.save(model.state_dict(), "rubert_model/pytorch_model.bin")
    tokenizer.save_pretrained("rubert_model")
    
    # Save config
    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "threshold": 0.5,
        "metrics": {
            "f1": float(f1_score(all_labels, all_preds)),
            "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        }
    }
    with open("rubert_model/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n? RuBERT сохранён в rubert_model/")
    
    return model, tokenizer, all_probs


# ==========================================
# ВАРИАНТ 3: Лёгкий fine-tuning (freeze encoder)
# ==========================================

def train_rubert_light(X_train, y_train, X_test, y_test):
    """Light fine-tuning: заморозка BERT, обучение только head."""
    print("\n" + "="*60)
    print("LIGHT FINE-TUNING: Заморозка RuBERT, обучаем только head")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = RuBERTClassifier(MODEL_NAME).to(DEVICE)
    
    # Freeze BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False
    
    print("Заморожено параметров BERT. Обучаем только классификационную голову.")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
    
    # Eval
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    
    print(f"\nAccuracy: {(all_preds == np.array(all_labels)).mean():.3f}")
    print(f"F1-score: {f1_score(all_labels, all_preds):.3f}")
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.3f}")
    
    return model, tokenizer, all_probs


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("RuBERT Fine-tuning для анализа суицидальных текстов")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n[1/4] Загрузка данных...")
    df = create_demo_dataset(samples_per_class=500)
    X_train, X_test, y_train, y_test = create_train_test_split(df, target_col='binary_label')
    
    # 2. Baseline (LogReg на эмбеддингах)
    print("\n[2/4] Baseline: Logistic Regression...")
    clf, X_train_emb, X_test_emb, logreg_proba = train_baseline_logreg(X_train, y_train, X_test, y_test)
    
    # 3. Fine-tuning RuBERT
    print("\n[3/4] Fine-tuning RuBERT...")
    rubert_model, rubert_tokenizer, rubert_proba = train_rubert_finetune(X_train, y_train, X_test, y_test)
    
    # 4. Сравнение
    print("\n" + "="*60)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*60)
    print(f"{'Модель':<30} {'ROC-AUC':>10} {'F1':>10}")
    print("-" * 60)
    
    logreg_f1 = f1_score(y_test, (logreg_proba >= 0.5).astype(int))
    logreg_auc = roc_auc_score(y_test, logreg_proba)
    print(f"{'LogReg на RuBERT [CLS]':<30} {logreg_auc:>10.3f} {logreg_f1:>10.3f}")
    
    rubert_f1 = f1_score(y_test, (np.array(rubert_proba) >= 0.5).astype(int))
    rubert_auc = roc_auc_score(y_test, rubert_proba)
    print(f"{'Fine-tuned RuBERT':<30} {rubert_auc:>10.3f} {rubert_f1:>10.3f}")
    
    print("="*60)
    print("\n? Готово! Модель сохранена в rubert_model/")
