import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import pipeline
import datetime
import os
import time
import torch
import traceback

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
    # Load the model once at the beginning
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        dtype=torch_dtype,
        chunk_length_s=SAMPLE_RATE,
        ignore_warning=True,
    )
    print("Model loaded. Ready to take notes.")

    try:
        while True:
            input("\nPress Enter to start a new note session (or Ctrl+C to exit)...")

            language = get_language_choice()

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
                with open(session_filename, "a") as f:
                    while True:
                        # 1. Record a chunk of audio
                        print(f"Recording a {CHUNK_SECONDS}-second chunk...")
                        audio_chunk = sd.rec(
                            int(CHUNK_SECONDS * SAMPLE_RATE),
                            samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=np.float32,
                        )
                        sd.wait()  # Wait for the recording to complete

                        # 2. Save the chunk to a temporary WAV file
                        temp_audio_file = "temp_chunk.wav"
                        write(
                            temp_audio_file, SAMPLE_RATE, audio_chunk
                        )

                        # 3. Transcribe the chunk
                        print("Transcribing chunk...")
                        result = transcriber(
                            temp_audio_file,
                            generate_kwargs={
                                "language": language,
                                "task": "transcribe",
                            },
                        )
                        transcription = result["text"]

                        # 4. Append to the file and print to the console
                        if transcription.strip():
                            clean_text = transcription.strip()
                            print(f"  -> Appending: '{clean_text}'")
                            f.write(clean_text + "\n")
                            f.flush()
                        else:
                            print("  -> No speech detected in this chunk.")

                        # 5. Clean up the temporary file
                        os.remove(temp_audio_file)

            except KeyboardInterrupt:
                print(
                    f"\n\nNote session finished. Your note is saved in {session_filename}"
                )
                print("--------------------------------------------------")
                pass

    except KeyboardInterrupt:
        print("\nExiting note taker. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
