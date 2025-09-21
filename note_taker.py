import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import (
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
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
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch_dtype
    ).to(device)
    print("Model loaded. Ready to take notes.")

    print("Loading the formatting LLM...")
    hf_token = os.getenv("HUGGING_FACE")
    llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=hf_token)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        token=hf_token,
        device_map=device,
        load_in_4bit=True if device == "cuda" else False,  # 4-bit still good for CUDA
        dtype=torch_dtype,
    )
    print("Formatting LLM loaded.")

    transcript_buffer = []
    raw_transcript_buffer = []
    full_formatted_text = ""

    vad = load_silero_vad()

    try:
        while True:
            # input("\nPress Enter to start a new note session (or Ctrl+C to exit)...")

            # language = get_language_choice()
            language = "pl"

            llm_call_count = 0
            llm_total_time = 0.0

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
                    with sd.InputStream(
                        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32
                    ) as stream:
                        vad_iterator = VADIterator(
                            vad, sampling_rate=16000, threshold=0.3
                        )
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
                            if (
                                speech_segments is not None
                                and "start" in speech_segments
                            ):
                                in_speech = True
                                # add some silence
                                buffer.extend(np.zeros(chunk_size) for _ in range(3))
                                # add a bit of pre-padding of original audio
                                if prepad_buffer:
                                    buffer.append(prepad_buffer[-1])

                            if in_speech:
                                buffer.append(data)
                            else:
                                prepad_buffer.append(data)

                            current_frame += data.shape[0]

                            if speech_segments is not None and "end" in speech_segments:
                                in_speech = False

                                # TODO: remove non-speech from the end

                                chunk = np.concatenate(buffer)

                                filename = f"chunks/{current_frame:08d}.wav"
                                write(
                                    filename,
                                    SAMPLE_RATE,
                                    chunk.reshape((chunk.shape[0], 1)),
                                )
                                print(f"produced a chunk: {chunk.shape} {filename}")

                                inputs = processor(
                                    chunk,
                                    sampling_rate=SAMPLE_RATE,
                                    return_tensors="pt",
                                ).to(device, dtype=torch_dtype)

                                # run generate with forced start tokens
                                with torch.no_grad():
                                    generated_ids = model.generate(
                                        **inputs,
                                        language=language,
                                        return_timestamps=True,
                                    )

                                tokens = generated_ids[0].tolist()
                                decoded = processor.tokenizer.convert_ids_to_tokens(
                                    tokens
                                )
                                print(decoded)

                                # decode to text
                                text = processor.batch_decode(
                                    generated_ids, skip_special_tokens=True
                                )[0]

                                raw_transcript_buffer.append(text)

                                # --- LLM Formatting ---
                                messages = [
                                    {
                                        "role": "system",
                                        "content": "You are a note-taking assistant. Your task is to format the raw, streaming transcript into a clean, coherent document. Retain the original language. Output only the formatted document.",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"""Here is the document so far:
{full_formatted_text}

Here is the new segment of the transcript:
{' '.join(raw_transcript_buffer)}

Please integrate the new segment into the document, correcting any previous formatting if the new context requires it. The output should be the *entire*, updated document. Do not add any conversational text or extra commentary.
""",
                                    },
                                ]

                                prompt = llm_tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )

                                inputs = llm_tokenizer(prompt, return_tensors="pt").to(
                                    device
                                )

                                start_time = time.monotonic()
                                with torch.no_grad():
                                    output_ids = llm_model.generate(
                                        **inputs,
                                        max_new_tokens=1024,
                                        do_sample=False,
                                        eos_token_id=llm_tokenizer.eos_token_id,
                                    )
                                end_time = time.monotonic()
                                llm_total_time += end_time - start_time
                                llm_call_count += 1

                                response_ids = output_ids[0][
                                    inputs["input_ids"].shape[-1] :
                                ]
                                full_formatted_text = llm_tokenizer.decode(
                                    response_ids, skip_special_tokens=True
                                ).strip()

                                # Overwrite the file with the latest formatted version
                                with open(session_filename, "w") as f:
                                    f.write(full_formatted_text)

                                print("--- Formatted Note ---")
                                print(full_formatted_text)
                                print("----------------------")

                                # Clear the raw buffer for the next chunk
                                raw_transcript_buffer.clear()

            except KeyboardInterrupt:
                print(
                    f"\n\nNote session finished. Your note is saved in {session_filename}"
                )
                print("--- LLM Performance ---")
                if llm_call_count > 0:
                    average_time = llm_total_time / llm_call_count
                    print(f"LLM calls: {llm_call_count}")
                    print(f"Total LLM time: {llm_total_time:.2f} seconds")
                    print(f"Average time per call: {average_time:.2f} seconds")
                else:
                    print("No LLM calls were made.")
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
