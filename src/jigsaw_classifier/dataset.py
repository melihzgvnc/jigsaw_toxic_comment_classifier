import torch
import string
from torch.utils.data import Dataset
import pandas as pd

from jigsaw_classifier.vocab import tokenizer

def transform(seq):
    words = seq.lower().split()
    translator = str.maketrans('', '', string.punctuation)
    words_cleaned = [word.translate(translator) for word in words]
    encoded_seq = tokenizer.encode(words_cleaned)
    return encoded_seq

def target_transform(df):
    label_cols = df.columns[2:]
    return df[label_cols].values.tolist()
    

class JigsawDataset(Dataset):
    def __init__(self, data_path=None, data_size=6000):
        data = pd.read_csv(data_path)
        data = data.sample(n=data_size, random_state=42)
        self.samples = data['comment_text']
        self.labels = target_transform(data)
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = torch.tensor(transform(self.samples.iloc[idx]))
        labels = torch.tensor(self.labels[idx])

        return (sample, labels, len(sample)) # token_ids, target_ids, seq_len

