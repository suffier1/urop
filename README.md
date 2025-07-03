# Multimodal Speech Emotion Recognition

This project provides simple scripts to train and test a multimodal emotion recognition model using the [AI Hub speech emotion dataset](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=637).  All audio files should be denoised beforehand.  The code automatically parses the `audiodata*` folders and corresponding `emotion_data*.csv` files.

## Dataset preparation

Place the `audiodata1`, `audiodata2` and `audiodata3` folders and their matching `emotion_data*.csv` files under a single directory, e.g. `data/`.  All wav files should already be denoised.
The code will automatically read the CSV files and split the dataset into train and validation sets.

## Training

Install dependencies (Python 3.12 recommended):

```bash
pip install torch transformers datasets pandas sounddevice faster-whisper tqdm noisereduce
```

Run training:

```bash
python train.py --data_root data --output_dir checkpoints
```

The script saves `model.pt` in the specified output directory.

## Testing interactively

After training, run the demo application:

```bash
python app.py --model checkpoints/model.pt --whisper_model small
```

1. Press `Ctrl+C` to stop recording.
2. The program prints the recognized text and the predicted emotion.

The demo records your voice, applies noise reduction and transcribes it using `faster-whisper` before predicting the emotion.
