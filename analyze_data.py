import pandas as pd
import os

file_path = 'c:/1 Sanjay/Political_SA/datas/PS_train.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Class Counts:")
    print(df['labels'].value_counts())
    print("\nUnique Labels:")
    print(df['labels'].unique())
else:
    print(f"File not found: {file_path}")
