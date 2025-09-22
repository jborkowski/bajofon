import sounddevice as sd
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import datetime
import os
import torch
import traceback
from collections import deque
import fire
from contextlib import ExitStack
import soundfile
import json
from dataclasses import dataclass
from typing import Optional

from silero_vad import load_silero_vad, VADIterator

# --- Configuration ---
MODEL_NAME = "openai/whisper-medium"
##MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
CHUNK_SECONDS = 30  # Duration of each audio chunk in seconds
OUTPUT_DIR = "notes"
SUPPORTED_LANGUAGES = {"pl", "en", "es"}

RECORDINGS_DIR = "recordings"


def get_language_choice():
    """Prompts the user to select a language and validates the input."""
    while True:
        lang = input(f"Choose a language ({', '.join(SUPPORTED_LANGUAGES)}): ").lower()
        if lang in SUPPORTED_LANGUAGES:
            return lang
        else:
            print(
                f"Invalid language. Please choose from: {', '.join(SUPPORTED_LANGUAGES)}"
            )

@dataclass
class SpeechSegment:
    start_frame: int
    chunks: list[np.ndarray]

def main(filename=None, input_audio_file=None, language='en'):
    """Main function to run the real-time note-taking application."""

    # --- Device and Data Type Configuration ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device.upper()} with data type: {torch_dtype}")

    print("Loading the Whisper model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch_dtype).to(device)
    print("Model loaded. Ready to take notes.")

    vad = load_silero_vad()

    try:

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recording_dir = f"{RECORDINGS_DIR}/{timestamp}"
        os.makedirs(recording_dir, exist_ok=True)

        if filename is None:
            # --- Create a new note file for the session ---
            filename = os.path.join(
                OUTPUT_DIR, f"note_{timestamp}_{language}.txt"
            )

        print(f"\nNew note session started. Language: {language.upper()}. Saving to: {filename}")
        print(f"Recording... Press Ctrl+C to stop and save.")

        print(f"Debug files at: {recording_dir}")

        with ExitStack() as stack:
            note_output = stack.enter_context(open(filename, "a"))
            mic_stream = stack.enter_context(sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32))
            original_audio_output = soundfile.SoundFile(f"{recording_dir}/original.flac", mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS, format='FLAC')
            voice_audio_output = soundfile.SoundFile(f"{recording_dir}/voice.flac", mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS, format='FLAC')
            voice_audio_current_frame = 0
            log_output = stack.enter_context(open(f"{recording_dir}/log.jsonl", "w"))

            current_frame = 0

            def log(message: dict):
                message['time'] = datetime.datetime.now().isoformat()
                print(json.dumps(message))
                log_output.write(f"{json.dumps(message)}\n")
                log_output.flush()

            log({"event": "start", "language": language, "model": MODEL_NAME, "filename": filename})

            chunk_size = 512
            min_silence_duration = 0.5  # seconds
            min_silence_chunks = int(min_silence_duration * SAMPLE_RATE / chunk_size)

            vad_iterator = VADIterator(vad, sampling_rate=16000, threshold=0.3, min_silence_duration_ms=int(min_silence_duration * 1000))
            max_chunks_for_whisper = SAMPLE_RATE * 30 // chunk_size

            prepad_buffer = deque(maxlen=SAMPLE_RATE // chunk_size)
            current_speech_segment: Optional[SpeechSegment] = None
            speech_segments = deque()

            while not mic_stream.closed:
                data, _ = mic_stream.read(chunk_size)
                data = data.squeeze()

                original_audio_output.write(data)

                vad_result = vad_iterator(data)
                if vad_result is not None and 'start' in vad_result:
                    log({"event": "speech_start", "start_frame": current_frame})
                    current_speech_segment = SpeechSegment(start_frame=current_frame, chunks=[])

                    # add a bit of pre-padding of original audio
                    if len(prepad_buffer) > 0:
                        num_prepad_chunks = min(5, len(prepad_buffer))
                        current_speech_segment.chunks.extend(list(prepad_buffer)[-num_prepad_chunks:])


                if current_speech_segment is not None:
                    current_speech_segment.chunks.append(data)
                else:
                    prepad_buffer.append(data)

                current_frame += data.shape[0]

                if current_speech_segment is not None and vad_result is not None and 'end' in vad_result:
                    log({"event": "speech_end", "end_frame": current_frame})

                    # TODO: remove non-speech from the end
                    # TODO: handle speech chunks longer than 30 seconds

                    # add some silence
                    current_speech_segment.chunks.extend([np.zeros(chunk_size) for _ in range(min_silence_chunks)])
                    speech_segments.append(current_speech_segment)
                    voice_audio_output.write(np.concatenate(current_speech_segment.chunks))
                    voice_audio_current_frame += sum(len(s) for s in current_speech_segment.chunks)
                    current_speech_segment = None

                    while sum(len(s.chunks) for s in speech_segments) > max_chunks_for_whisper:
                        segment = speech_segments.popleft()
                        log({"event": "segment_dropped", "start_frame": segment.start_frame, "num_chunks": len(segment.chunks)})

                    chunk = np.concatenate([s for s in speech_segments for s in s.chunks])

                    inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True).to(device, dtype=torch_dtype)

                    # run generate with forced start tokens
                    with torch.no_grad():
                        result = model.generate(
                            **inputs,
                            task="transcribe",
                            language=language,
                            return_dict_in_generate=True,
                            return_timestamps=True,
                            return_token_timestamps=True,
                        )

                    print(list(result.keys()))

                    generated_ids = result["sequences"]
                 
                    # decode to text
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True, language=language)[0]

                    print()
                    print(text)
                    print(result["token_timestamps"])
                    log({"event": "transcription", "text": text})

                    note_output.write(f"{text}\n")
                    note_output.flush()


    except KeyboardInterrupt:
        print("\nExiting note taker. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)
