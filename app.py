import argparse
import queue
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer
import noisereduce as nr

from model import AudioTextEmotionModel

SAMPLE_RATE = 16000


def record_audio(duration=None):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        print('Recording... Press Ctrl+C to stop.')
        frames = []
        try:
            while True:
                frames.append(q.get())
                if duration and len(frames) * 1024 / SAMPLE_RATE >= duration:
                    break
        except KeyboardInterrupt:
            print('Recording stopped.')
    audio = np.concatenate(frames, axis=0).squeeze()
    audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
    return audio


def recognize_text(audio, model):
    segments, _ = model.transcribe(audio, language='ko')
    return ''.join(seg.text for seg in segments)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = AudioTextEmotionModel(args.text_model, args.audio_model, num_labels=len(checkpoint['label2id']))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    label2id = checkpoint['label2id']
    id2label = {v: k for k, v in label2id.items()}
    stt_model = WhisperModel(args.whisper_model, device=device)

    while True:
        audio = record_audio()
        waveform = torch.tensor(audio, dtype=torch.float32)
        text = recognize_text(audio, stt_model)
        print('Transcription:', text)

        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            output = model(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                input_values=waveform.unsqueeze(0).to(device)
            )
            pred = output['logits'].argmax(dim=1).item()
            print('Predicted emotion:', id2label[pred])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text_model', default='distilbert-base-uncased')
    parser.add_argument('--audio_model', default='facebook/wav2vec2-base-960h')
    parser.add_argument('--whisper_model', default='small')
    args = parser.parse_args()
    main(args)
