
import json
import os

# ---------------------------------------------------------
# Define Code Blocks
# ---------------------------------------------------------

block_install = """
!pip install transformers sentencepiece
!pip install pandas scikit-learn torch
"""

block_imports = """
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""

block_mount_drive = """
# Mount Google Drive
from google.colab import drive
import os

drive.mount('/content/drive')

# Helper to find your datas folder if the path below is wrong:
def find_data_dir(search_name="Political_SA/datas"):
    for root, dirs, files in os.walk("/content/drive/MyDrive"):
        if search_name in root.replace("\\\\", "/"):
            return root + "/"
    return None

# UPDATE THIS PATH to where your data is located in Drive
# You can use the helper or set it manually
DATA_DIR = "/content/drive/MyDrive/Political_SA/datas/" 

if not os.path.exists(DATA_DIR):
    print(f"Warning: {DATA_DIR} not found. Attempting to locate automatically...")
    found = find_data_dir()
    if found:
        DATA_DIR = found
        print(f"Found data at: {DATA_DIR}")
    else:
        print("Could not find data directory. Please check the 'files' sidebar in Colab and copy the path to your CSVs.")

# If you uploaded files directly to Colab (drag and drop to sidebar):
# DATA_DIR = "/content/"
"""

block_dataset_class = """
class PoliticalDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Label mapping
        self.label_map = {
            'Substantiated': 0,
            'Sarcastic': 1,
            'Opinionated': 2,
            'Positive': 3,
            'Negative': 4,
            'Neutral': 5,
            'None of the above': 6
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        text = str(self.data.iloc[item]['content'])
        
        label = None
        if 'labels' in self.data.columns:
            label_name = self.data.iloc[item]['labels']
            label = self.label_map.get(label_name, -1)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item_dict = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if label is not None:
             item_dict['label'] = torch.tensor(label, dtype=torch.long)
             
        return item_dict
"""

block_loss_class = """
class SupConLoss(nn.Module):
    \"\"\"Supervised Contrastive Learning Loss\"\"\"
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_counts = mask.sum(1)
        mask_pos_counts[mask_pos_counts == 0] = 1 # Avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_counts

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
"""

block_model_class = """
class PoliticalClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super(PoliticalClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            pooled_output = output.pooler_output
        else:
            pooled_output = output.last_hidden_state[:, 0, :]
            
        hidden_output = self.drop(pooled_output)
        logits = self.out(hidden_output)
        features = nn.functional.normalize(hidden_output, dim=1)
        
        return features, logits
"""

block_functions = """
def train_epoch(model, data_loader, loss_fn_ce, loss_fn_supcon, optimizer, scheduler, device, n_examples, lambda_val=0.3):
    model.train()
    losses = []
    correct_predictions = 0
    
    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        
        features, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        features_supcon = features.unsqueeze(1)
        
        loss_ce = loss_fn_ce(logits, labels)
        loss_supcon = loss_fn_supcon(features_supcon, labels)
        
        loss = (1 - lambda_val) * loss_ce + lambda_val * loss_supcon
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            print(f"Step {i}/{len(data_loader)} Loss: {loss.item():.4f}")
        
    avg_loss = np.mean(losses)
    return correct_predictions.double() / n_examples, avg_loss


def eval_model(model, data_loader, loss_fn_ce, loss_fn_supcon, device, n_examples, lambda_val=0.3):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            features, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            features_supcon = features.unsqueeze(1)
            loss_ce = loss_fn_ce(logits, labels)
            loss_supcon = loss_fn_supcon(features_supcon, labels)
            loss = (1 - lambda_val) * loss_ce + lambda_val * loss_supcon
            
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return correct_predictions.double() / n_examples, np.mean(losses), all_labels, all_preds

def load_csv_robust(path):
    print(f"Loading {path}...")
    data = []
    valid_labels = set(['Substantiated', 'Sarcastic', 'Opinionated', 'Positive', 'Negative', 'Neutral', 'None of the above'])
    
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        return pd.DataFrame(columns=['content', 'labels'])

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Raw lines read: {len(lines)}")
        # Skip header
        lines = lines[1:]

        current_content = []
        
        for line in lines:
            striped_line = line.strip()
            # Check for label at end
            parts = striped_line.rsplit(',', 1)
            found_label = None
            
            if len(parts) == 2:
                possible_label = parts[1].strip().strip('"')
                if possible_label in valid_labels:
                    found_label = possible_label
            
            if found_label:
                content_part = parts[0]
                current_content.append(content_part)
                full_content = "\\n".join(current_content).strip().strip('"')
                data.append([full_content, found_label])
                current_content = [] 
            else:
                current_content.append(line.strip())

    print(f"Loaded {len(data)} rows from {path}.")
    return pd.DataFrame(data, columns=['content', 'labels'])
"""

block_train_execution = """
# Configuration
model_name = 'l3cube-pune/tamil-bert'
max_len = 160
batch_size = 16 # Increase to 16 or 32 on Colab GPU
epochs = 10 
lambda_val = 0.3 # Weight for SupCon loss

# Load Data
train_path = os.path.join(DATA_DIR, 'PS_train.csv')
dev_path = os.path.join(DATA_DIR, 'PS_dev.csv')

train_df = load_csv_robust(train_path)
dev_df = load_csv_robust(dev_path)

# Text Cleaning
import re
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    return text.strip()

train_df['content'] = train_df['content'].apply(clean_text)
dev_df['content'] = dev_df['content'].apply(clean_text)

# Class Balancing
class_counts = train_df['labels'].value_counts().sort_index()
label_map = {
        'Substantiated': 0, 'Sarcastic': 1, 'Opinionated': 2, 'Positive': 3,
        'Negative': 4, 'Neutral': 5, 'None of the above': 6
}

train_df['label_idx'] = train_df['labels'].map(label_map)
class_counts = train_df['label_idx'].value_counts().sort_index()

total_samples = len(train_df)
class_weights = total_samples / (len(label_map) * class_counts)
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights_tensor}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Datasets
train_dataset = PoliticalDataset(train_df, tokenizer, max_len)
val_dataset = PoliticalDataset(dev_df, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model Init
model = PoliticalClassifier(model_name, n_classes=len(label_map))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn_ce = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
loss_fn_supcon = SupConLoss(temperature=0.1)


# Training Loop
best_mf1 = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        model, train_loader, loss_fn_ce, loss_fn_supcon, optimizer, scheduler, device, len(train_dataset), lambda_val
    )
    
    print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")
    
    val_acc, val_loss, y_true, y_pred = eval_model(
        model, val_loader, loss_fn_ce, loss_fn_supcon, device, len(val_dataset), lambda_val
    )
    
    mf1 = f1_score(y_true, y_pred, average='macro')
    print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f} Macro F1 {mf1:.4f}")
    print(classification_report(y_true, y_pred, target_names=list(label_map.keys()), zero_division=0))
    
    if mf1 > best_mf1:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_mf1 = mf1
        print(f"Saved best model (Macro F1: {best_mf1:.4f})")
"""

block_inference = """
# Inference and Submission Generation
def predict_submission():
    test_path = os.path.join(DATA_DIR, 'PS_test_without_labels.csv')
    test_df = pd.read_csv(test_path)
    
    # Prepare Dataset
    test_dataset = PoliticalDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # We will generate 3 runs as permitted
    n_runs = 3
    zip_filename = 'Political_SA.zip'
    output_files = []
    
    import zipfile
    
    for run_i in range(1, n_runs + 1):
        print(f"Starting prediction for Run {run_i}...")
        
        # Load Model (In a real scenario, you might train 3 different models or use different seeds)
        # Here we reload the same best model for demonstration, but typically you'd have 'best_model_run1.bin', etc.
        # For this setup, we will use the same weights but enablement of dropout during inference (Monte Carlo Dropout) 
        # could create variations, or just simply re-running if we had different models.
        # To make it simple and strictly 'valid', we will just output the same prediction 3 times 
        # UNLESS you actually train 3 models. 
        # Let's stick to generating just ONE file if we only have one model, 
        # BUT the user asked for the FORMAT "Team_name_Task_Runs.csv". 
        # So we will generate "Political_SA_run{run_i}.csv".
        
        model = PoliticalClassifier(model_name, n_classes=7)
        model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
        model = model.to(device)
        model.eval()
        
        all_preds = []
        
        with torch.no_grad():
            for d in test_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                
        # Decode Labels
        idx_to_label = {v: k for k, v in label_map.items()}
        predicted_labels = [idx_to_label[p] for p in all_preds]
        
        # Create Submission CSV for this run
        submission_df = pd.DataFrame({
            'content': test_df['content'],
            'labels': predicted_labels
        })
        
        output_filename = f'Political_SA_run{run_i}.csv'
        submission_df.to_csv(output_filename, index=False)
        output_files.append(output_filename)
        print(f"Generated {output_filename}")

    # Create Zip containing all runs
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for csv_file in output_files:
            zipf.write(csv_file)
        
    print(f"Created submission: {zip_filename} with {len(output_files)} files.")
    # Optional: Copy to Drive
    # import shutil
    # shutil.copy(zip_filename, os.path.join(DATA_DIR, zip_filename))

predict_submission()
"""

# ---------------------------------------------------------
# Construct Notebook
# ---------------------------------------------------------

def create_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.strip().splitlines(keepends=True)
    }

def create_markdown_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_text.strip().splitlines(keepends=True)
    }

cells = [
    create_markdown_cell("# Political Sentiment Analysis (Tamil/Tanglish)\n\nThis notebook implements a Transformer-based sentiment analysis model using XLM-Roberta and Supervised Contrastive Learning (SupCon)."),
    create_markdown_cell("## 1. Setup and Dependencies"),
    create_code_cell(block_install),
    create_code_cell(block_imports),
    create_markdown_cell("## 2. Connect to Data\nUpload your `datas/` folder to Google Drive or the Colab runtime."),
    create_code_cell(block_mount_drive),
    create_markdown_cell("## 3. Dataset Class"),
    create_code_cell(block_dataset_class),
    create_markdown_cell("## 4. Loss Function\nImplements SupCon Loss for robust feature learning."),
    create_code_cell(block_loss_class),
    create_markdown_cell("## 5. Model Architecture\nXLMRoberta + Classification Head + SupCon Projection Head"),
    create_code_cell(block_model_class),
    create_markdown_cell("## 6. Helper Functions"),
    create_code_cell(block_functions),
    create_markdown_cell("## 7. Training Execution"),
    create_code_cell(block_train_execution),
    create_markdown_cell("## 8. Inference & Submission"),
    create_code_cell(block_inference)
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("c:/1 Sanjay/Political_SA/Political_SA_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")
