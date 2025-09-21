import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import datetime
import os
import time
import torch
import traceback
from collections import deque

from silero_vad import load_silero_vad, VADIterator

# --- Configuration ---
MODEL_NAME = "openai/whisper-medium"
##MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
CHUNK_SECONDS = 30  # Duration of each audio chunk in seconds
OUTPUT_DIR = "notes"
SUPPORTED_LANGUAGES = {"pl", "en", "es"}


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


def main():
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
        while True:
            # input("\nPress Enter to start a new note session (or Ctrl+C to exit)...")

            #language = get_language_choice()
            language = 'pl'

            # --- Create a new note file for the session ---
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_filename = os.path.join(
                OUTPUT_DIR, f"note_{timestamp}_{language}.txt"
            )

            print(
                f"\nNew note session started. Language: {language.upper()}. Saving to: {session_filename}"
            )
            print(
                f"Recording in {CHUNK_SECONDS}-second chunks... Press Ctrl+C to stop and save."
            )

            try:
                with open(session_filename, "a"):
                    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32) as stream:
                        vad_iterator = VADIterator(vad, sampling_rate=16000, threshold=0.3)
                        chunk_size = 512
                        max_chunks_for_whisper = SAMPLE_RATE * 30 // chunk_size

                        buffer = deque(maxlen=max_chunks_for_whisper)
                        prepad_buffer = deque(maxlen=SAMPLE_RATE // chunk_size)
                        in_speech = False

                        current_frame = 0
                        while not stream.closed:
                            data, _ = stream.read(chunk_size)
                            data = data.squeeze()

                            speech_segments = vad_iterator(data)
                            if speech_segments is not None and 'start' in speech_segments:
                                in_speech = True
                                # add some silence
                                buffer.extend(np.zeros(chunk_size) for _ in range(3))
                                # add a bit of pre-padding of original audio
                                buffer.append(prepad_buffer[-1])

                            if in_speech:
                                buffer.append(data)
                            else:
                                prepad_buffer.append(data)

                            current_frame += data.shape[0]

                            if speech_segments is not None and 'end' in speech_segments:
                                in_speech = False

                                # TODO: remove non-speech from the end

                                chunk = np.concatenate(buffer)

                                filename = f"chunks/{current_frame:08d}.wav"
                                write(filename, SAMPLE_RATE, chunk.reshape((chunk.shape[0], 1)))
                                print(f"produced a chunk: {chunk.shape} {filename}")

                                inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device, dtype=torch_dtype)

                                # run generate with forced start tokens
                                with torch.no_grad():
                                    generated_ids = model.generate(
                                            **inputs,
                                            #   decoder_input_ids=prompt_ids
                                            return_timestamps=True
                                            )

                                tokens = generated_ids[0].tolist()
                                decoded = processor.tokenizer.convert_ids_to_tokens(tokens)
                                print(decoded)
                             
                                # decode to text
                                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                                print(text)


            except KeyboardInterrupt:
                print(
                    f"\n\nNote session finished. Your note is saved in {session_filename}"
                )
                print("--------------------------------------------------")
                raise
                pass

    except KeyboardInterrupt:
        print("\nExiting note taker. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
