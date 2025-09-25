import argparse
import queue
import sys
from typing import Generator, Optional

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SAMPLE_RATE = 16000


def stream_audio(chunk_duration: float, energy_threshold: float) -> Generator[np.ndarray, None, None]:
    """Yield audio chunks of roughly ``chunk_duration`` seconds."""
    q: "queue.Queue[np.ndarray]" = queue.Queue()
    chunk_samples = int(chunk_duration * SAMPLE_RATE)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=callback,
    ):
        buffer = np.empty((0, 1), dtype=np.float32)
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                data = q.get()
                buffer = np.concatenate([buffer, data], axis=0)
                while len(buffer) >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
                    if np.mean(np.abs(chunk)) < energy_threshold:
                        continue
                    yield chunk[:, 0]
        except KeyboardInterrupt:
            print("Stopping listener.")
            return


def build_translator(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def translate(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    return translate


def transcribe_and_translate(
    stt_model: WhisperModel,
    translate_fn,
    audio_chunk: np.ndarray,
    source_language: Optional[str],
    task: str,
) -> Optional[tuple[str, str]]:
    language = source_language if source_language and source_language.lower() != "auto" else None
    segments, _ = stt_model.transcribe(
        audio_chunk,
        language=language,
        task=task,
        beam_size=5,
        vad_filter=True,
    )
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        return None

    translation = translate_fn(text)
    return text, translation


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_model = WhisperModel(args.whisper_model, device=whisper_device, compute_type=args.compute_type)
    translate_fn = build_translator(args.translation_model, device)

    try:
        for chunk in stream_audio(args.chunk_duration, args.energy_threshold):
            result = transcribe_and_translate(
                stt_model,
                translate_fn,
                chunk,
                args.source_language,
                task="transcribe",
            )
            if result is None:
                continue
            original, translated = result
            print("\n[Original]", original)
            print("[Translated]", translated)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time speech translation demo.")
    parser.add_argument("--whisper_model", default="small", help="Name of the faster-whisper model to use.")
    parser.add_argument(
        "--translation_model",
        default="Helsinki-NLP/opus-mt-ko-en",
        help="Sequence-to-sequence model for text translation.",
    )
    parser.add_argument("--source_language", default="ko", help="Source language code or 'auto' for detection.")
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=5.0,
        help="Audio chunk duration in seconds for incremental transcription.",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=1e-3,
        help="Skip chunks whose average absolute amplitude falls below this threshold.",
    )
    parser.add_argument(
        "--compute_type",
        default="float32",
        help="faster-whisper compute type (e.g., float16, int8).",
    )
    args = parser.parse_args()
    main(args)
