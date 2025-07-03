import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import noisereduce as nr


class EmotionDataset(Dataset):
    """Dataset for AIHub emotional speech dataset with denoising."""

    def __init__(self, data_root, split='train', valid_ratio=0.1, tokenizer=None,
                 sample_rate=16000, max_length=128, label2id=None):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.tokenizer = tokenizer

        # load all csv files
        frames = []
        for i in (1, 2, 3):
            csv_name = f"emotion_data{i}.csv"
            csv_path = os.path.join(data_root, csv_name)
            if not os.path.exists(csv_path):
                # some files may be named slightly differently
                csv_name = f"emotion__data{i}.csv"
                csv_path = os.path.join(data_root, csv_name)
            df = pd.read_csv(csv_path, sep='\t')
            audio_dir = os.path.join(data_root, f"audiodata{i}")
            df['path'] = df['wav_id'].apply(lambda x: os.path.join(audio_dir, f"{x}.wav"))
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        # choose label by highest intensity
        emotions = df[["1번 감정", "2번 감정", "3번 감정", "4번 감정", "5번 감정"]].values
        strengths = df[["1번 감정세기", "2번 감정세기", "3번 감정세기", "4번 감정세기", "5번 감정세기"]].values
        labels = []
        for emo_row, str_row in zip(emotions, strengths):
            idx = int(pd.Series(str_row).astype(float).fillna(0).astype(float).idxmax())
            labels.append(emo_row[idx])
        df['text'] = df['발화문']
        df['label'] = labels
        df = df[['path', 'text', 'label']]
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(df) * (1 - valid_ratio))
        if split == 'train':
            self.data = df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.data = df.iloc[split_idx:].reset_index(drop=True)

        if label2id is None:
            labels = sorted(df['label'].unique())
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
        # apply noise reduction
        waveform = torch.tensor(nr.reduce_noise(y=waveform.numpy(), sr=self.sample_rate), dtype=torch.float32)

        text_inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors='pt')
        label_id = self.label2id[label]

        return {
            'input_values': waveform,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
