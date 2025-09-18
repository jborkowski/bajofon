import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import pipeline
import datetime
import os
import time
import torch

# --- Configuration ---
MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
CHUNK_SECONDS = 5  # Duration of each audio chunk in seconds
OUTPUT_DIR = "notes"


def main():
    """Main function to run the real-time note-taking application."""

    # --- Device and Data Type Configuration ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype_map = {
        "cpu": "float32",
        "mps": "float32",
        "cuda": "float16",
    }

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    numpy_dtype = dtype_map[device]

    print(f"Using device: {device.upper()} with data type: {numpy_dtype}")

    print("Loading the Whisper model...")
    # Load the model once at the beginning
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        dtype=torch_dtype,
    )
    print("Model loaded. Ready to take notes.")

    try:
        while True:
            input("\nPress Enter to start a new note session (or Ctrl+C to exit)...")

            # --- Create a new note file for the session ---
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_filename = os.path.join(OUTPUT_DIR, f"note_{timestamp}.txt")

            print(f"\nNew note session started. Saving to: {session_filename}")
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
                            dtype=numpy_dtype,
                        )
                        sd.wait()  # Wait for the recording to complete

                        # 2. Save the chunk to a temporary WAV file
                        temp_audio_file = "temp_chunk.wav"
                        # Ensure data is in a writable format (e.g., float32 or int16)
                        write(
                            temp_audio_file, SAMPLE_RATE, audio_chunk.astype(np.float32)
                        )

                        # 3. Transcribe the chunk
                        print("Transcribing chunk...")
                        result = transcriber(temp_audio_file)
                        transcription = result["text"]

                        # 4. Append to the file and print to the console
                        if transcription.strip():
                            print(f"  -> Appending: '{transcription.strip()}'")
                            f.write(transcription + " ")
                            f.flush()  # Ensure the text is written to the file immediately
                        else:
                            print("  -> No speech detected in this chunk.")

                        # 5. Clean up the temporary file
                        os.remove(temp_audio_file)

            except KeyboardInterrupt:
                print(
                    f"\n\nNote session finished. Your note is saved in {session_filename}"
                )
                print("--------------------------------------------------")
                # This allows the outer loop to continue for a new session
                pass

    except KeyboardInterrupt:
        print("\nExiting note taker. Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
