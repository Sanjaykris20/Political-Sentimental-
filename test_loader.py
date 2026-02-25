# -*- coding: utf-8 -*-
import pandas as pd
import time

def load_csv_robust(path):
    print(f"Starting to load {path}...")
    start_time = time.time()
    data = []
    skipped = 0
    valid_labels = set(['Substantiated', 'Sarcastic', 'Opinionated', 'Positive', 'Negative', 'Neutral', 'None of the above'])
    
    with open(path, 'r', encoding='utf-8') as f:
        # Skip header if present
        first_line = f.readline()
        if not any(lbl in first_line for lbl in valid_labels):
                pass
        else:
                f.seek(0)
                
        f.seek(0)
        lines = f.readlines()
        print(f"Read {len(lines)} raw lines. Processing...")
        
        lines = lines[1:] # Skip header assumption

        current_content = []
        
        for i, line in enumerate(lines):
            if i % 5000 == 0:
                print(f"Processing line {i}...")
            
            striped_line = line.strip()
            # Check if line ends with a valid label
            found_label = None
            
            # Check for ",Label" or ", Label"
            # Split by last comma
            parts = striped_line.rsplit(',', 1)
            if len(parts) == 2:
                possible_label = parts[1].strip().strip('"')
                if possible_label in valid_labels:
                    found_label = possible_label
            
            if found_label:
                # End of an entry
                content_part = parts[0]
                current_content.append(content_part)
                full_content = "\n".join(current_content).strip().strip('"')
                
                data.append([full_content, found_label])
                current_content = [] # Reset buffer
            else:
                # Continuation
                current_content.append(line.strip())

    print(f"Loaded {len(data)} rows from {path}. Skipped {skipped} rows.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return pd.DataFrame(data, columns=['content', 'labels'])

if __name__ == "__main__":
    df = load_csv_robust('c:/1 Sanjay/Political_SA/datas/PS_train.csv')
    print(df.head())
    print(df['labels'].value_counts())
