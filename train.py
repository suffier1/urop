import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from emotion_dataset import EmotionDataset
from model import AudioTextEmotionModel


def collate_fn(batch):
    audios = [b['input_values'] for b in batch]
    input_ids = [b['input_ids'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    labels = torch.stack([b['labels'] for b in batch])

    audio_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_values': audio_padded,
            'input_ids': ids_padded,
            'attention_mask': mask_padded,
            'labels': labels}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_ds = EmotionDataset(args.data_root, split='train', valid_ratio=args.valid_ratio, tokenizer=tokenizer)
    valid_ds = EmotionDataset(args.data_root, split='valid', valid_ratio=args.valid_ratio, tokenizer=tokenizer, label2id=train_ds.label2id)
    num_labels = len(train_ds.label2id)

    model = AudioTextEmotionModel(args.text_model, args.audio_model, num_labels)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Training {epoch}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Train loss: {avg_loss:.4f}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs['logits'].argmax(dim=1)
                correct += (preds == batch['labels']).sum().item()
                total += preds.size(0)
        acc = correct / total if total else 0
        print(f'Epoch {epoch} Val accuracy: {acc:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pt')
    torch.save({'model_state': model.state_dict(), 'label2id': train_ds.label2id}, model_path)
    print('Model saved to', model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Root directory containing audiodata folders and csv files')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Portion of data used for validation')
    parser.add_argument('--text_model', default='distilbert-base-uncased')
    parser.add_argument('--audio_model', default='facebook/wav2vec2-base-960h')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
