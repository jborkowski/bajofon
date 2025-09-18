import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import pipeline
import datetime
import os

# --- Configuration ---
MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
OUTPUT_DIR = "notes"

def record_audio():
    """Records audio from the microphone until the user presses Enter."""
    print("Press Enter to start recording...")
    input()
    print("Recording... Press Enter to stop.")
    
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        input() # Wait for Enter to be pressed again
    
    print("Recording finished.")
    return np.concatenate(audio_data, axis=0)

def save_audio_to_file(audio_data, filename="temp_recording.wav"):
    """Saves the recorded audio data to a WAV file."""
    write(filename, SAMPLE_RATE, audio_data)
    return filename

def transcribe_audio(filename):
    """Transcribes the audio file using the Whisper model."""
    print("Transcribing audio... (This may take a moment)")
    pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device="cpu")
    result = pipe(filename)
    return result["text"]

def save_transcription(text):
    """Saves the transcribed text to a timestamped file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"note_{timestamp}.txt")
    
    with open(filename, "w") as f:
        f.write(text)
        
    print(f"Transcription saved to: {filename}")
    return filename

def main():
    """Main function to run the note-taking application."""
    try:
        while True:
            audio_data = record_audio()
            temp_audio_file = save_audio_to_file(audio_data)
            
            transcription = transcribe_audio(temp_audio_file)
            
            if transcription.strip(): # Only save if there is transcribed text
                save_transcription(transcription)
            else:
                print("No speech detected, note not saved.")

            os.remove(temp_audio_file) # Clean up the temporary audio file
            
            print("\n-------------------\n")

    except KeyboardInterrupt:
        print("\nExiting note taker. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
