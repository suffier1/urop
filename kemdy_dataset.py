import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class KemdyDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label2id=None, max_length=128, sample_rate=16000):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_rate = sample_rate
        if label2id is None:
            labels = sorted(self.data['label'].unique())
            self.label2id = {l: i for i, l in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wav_path = row['path']
        text = row['text']
        label = row['label']

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)

        text_inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors='pt')
        label_id = self.label2id[label]

        return {
            'input_values': waveform,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
