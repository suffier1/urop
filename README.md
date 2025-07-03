# Multimodal Speech Emotion Recognition

This project provides simple scripts to train and test an emotion recognition model that uses both audio and text information. It targets the [KEMDy20](https://nanum.etri.re.kr/share/kjnoh2/KEMDy20?lang=En_us) dataset but can be adapted to any dataset organized as a CSV file.

## Dataset preparation

1. Download the KEMDy20 dataset manually from the ETRI website.
2. Create a CSV file with the following columns:
   - `path`: path to each WAV file.
   - `text`: transcript of the utterance.
   - `label`: emotion label (e.g., `angry`, `sad`, `neutral`, etc.).
3. Split the CSV into training and validation files, e.g. `train.csv` and `valid.csv`.

## Training

Install dependencies (Python 3.12 recommended):

```bash
pip install torch transformers datasets pandas sounddevice speechrecognition pocketsphinx tqdm
```

Run training:

```bash
python train.py --train_csv train.csv --valid_csv valid.csv --output_dir checkpoints
```

The script saves `model.pt` in the specified output directory.

## Testing interactively

After training, run the demo application:

```bash
python app.py --model checkpoints/model.pt
```

1. Press `Ctrl+C` to stop recording.
2. The program prints the recognized text and the predicted emotion.

The demo uses `pocketsphinx` for offline speech recognition. You can replace it with any preferred speech-to-text system by modifying `app.py`.
