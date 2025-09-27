import datetime
import json
import os
import socket
import threading
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Annotated, Optional, Literal, Union
from pydantic import BaseModel, Field, ValidationError

import fire
import numpy as np
import sounddevice as sd
import soundfile
import torch
from silero_vad import VADIterator, load_silero_vad
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# --- Configuration ---
DEFAULT_MODEL_NAME = "openai/whisper-medium"
# other options:
# "openai/whisper-large-v3"
# "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
OUTPUT_DIR = "notes"
SUPPORTED_LANGUAGES = {"pl", "en", "es"}

SOCKET_PATH = "/tmp/note_taker.sock"
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


class InsertTextCommand(BaseModel):
    """A command to insert a piece of text."""

    command: Literal["insert_text"]
    text: str


class ScreenshotCommand(BaseModel):
    """A command to insert screenshot"""

    command: Literal["screenshot"]
    path: str


Command = Annotated[
    InsertTextCommand | ScreenshotCommand, Field(discriminator="command")
]


def command_server_loop(command_queue: deque[Command], sock: socket.socket):
    """Accepts connections on the given socket and adds commands to a queue."""
    try:
        while True:
            connection, client_address = sock.accept()
            try:
                data = connection.recv(1024)
                if data:
                    try:
                        message = json.loads(data.decode("utf-8"))
                        command = Command.model_validate(message)
                        command_queue.append(command)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding command: {e}")
                    except ValidationError as e:
                        print(f"Invalid command received: {e}")
            finally:
                connection.close()
    except Exception:
        # Socket was likely closed by the main thread.
        print("Command server shutting down.")


@dataclass
class SpeechSegment:
    start_frame: int
    chunks: list[np.ndarray]
    transcriptions: list[str]

    def duration_seconds(self):
        return sum(len(c) for c in self.chunks) / SAMPLE_RATE

    def best_transcription(self) -> str:
        if len(self.transcriptions) == 0:
            return ""

        # return the last one
        return self.transcriptions[-1]

        # The code below seems smart, but doesn't work that well:

        # # count occurrences of each transcription
        # counts = {}
        # for t in self.transcriptions:
        #     t = t.strip()
        #     if t == "":
        #         continue
        #     if t not in counts:
        #         counts[t] = 0
        #     counts[t] += 1
        # # return the most common transcription
        # return max(counts.items(), key=lambda x: x[1])[0]


@dataclass
class CustomInputSegment:
    frame: int
    content: str


def main(
    filename=None, input_audio_file=None, language="en", model_name=DEFAULT_MODEL_NAME
):
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
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, dtype=torch_dtype
    ).to(device)
    print("Model loaded. Ready to take notes.")

    vad = load_silero_vad()

    # nlp = spacy.load("en_core_web_sm")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    recording_dir = f"{RECORDINGS_DIR}/{timestamp}"
    os.makedirs(recording_dir, exist_ok=True)

    if filename is None:
        # --- Create a new note file for the session ---
        filename = os.path.join(OUTPUT_DIR, f"note_{timestamp}_{language}.txt")

    print(
        f"\nNew note session started. Language: {language.upper()}. Saving to: {filename}"
    )
    print("Recording... Press Ctrl+C to stop and save.")

    print(f"Debug files at: {recording_dir}")

    with ExitStack() as stack:
        note_output = stack.enter_context(open(filename, "a"))
        mic_stream = stack.enter_context(
            sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)
        )
        original_audio_output = soundfile.SoundFile(
            f"{recording_dir}/original.flac",
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format="FLAC",
        )
        voice_audio_output = soundfile.SoundFile(
            f"{recording_dir}/voice.flac",
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format="FLAC",
        )
        voice_audio_current_frame = 0
        log_output = stack.enter_context(open(f"{recording_dir}/log.jsonl", "w"))

        # Command server setup
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stack.callback(server_socket.close)
        stack.callback(lambda: os.path.exists(SOCKET_PATH) and os.unlink(SOCKET_PATH))
        server_socket.bind(SOCKET_PATH)
        server_socket.listen(1)
        print(f"Command server listening on {SOCKET_PATH}")

        current_frame = 0

        def log(message: dict):
            message["time"] = datetime.datetime.now().isoformat()
            print(json.dumps(message))
            log_output.write(f"{json.dumps(message)}\n")
            log_output.flush()

        log(
            {
                "event": "start",
                "language": language,
                "model": model_name,
                "filename": filename,
            }
        )

        chunk_size = 512
        min_silence_duration = 0.5  # seconds
        min_silence_chunks = int(min_silence_duration * SAMPLE_RATE / chunk_size)

        vad_iterator = VADIterator(
            vad,
            sampling_rate=16000,
            threshold=0.3,
            min_silence_duration_ms=int(min_silence_duration * 1000),
        )
        max_chunks_for_whisper = SAMPLE_RATE * 30 // chunk_size

        prepad_buffer = deque(maxlen=SAMPLE_RATE // chunk_size)
        current_speech_segment: Optional[SpeechSegment] = None
        speech_segments = deque()
        custom_input_segments: deque[CustomInputSegment] = deque()
        command_queue: deque[Command] = deque()

        def drop_segment(segment: SpeechSegment):
            custom_input_segments_to_drop = 0
            for cs in custom_input_segments:
                if segment.start_frame > cs.frame:
                    log(
                        {
                            "event": "custom_input_attached",
                            "frame": cs.frame,
                            "content": cs.content,
                        }
                    )
                    note_output.write(f"{cs.content}\n")
                    custom_input_segments_to_drop += 1

            for _ in range(custom_input_segments_to_drop):
                custom_input_segments.popleft()

            chosen_transcription = segment.best_transcription()

            if chosen_transcription.strip().endswith((".", "!", "?")):
                chosen_transcription += "\n"

            print(f"OUT: {chosen_transcription}")

            note_output.write(f"{chosen_transcription}")
            note_output.flush()

            log(
                {
                    "event": "segment_dropped",
                    "start_frame": segment.start_frame,
                    "num_chunks": len(segment.chunks),
                    "transcriptions": segment.transcriptions,
                    "chosen_transcription": chosen_transcription,
                }
            )

        input_queue_mutex = threading.Lock()
        input_queue = deque()

        def custom_input_loop():
            while True:
                user_input = input()
                with input_queue_mutex:
                    input_queue.append(user_input)

        threading.Thread(target=custom_input_loop, daemon=True).start()

        command_thread = threading.Thread(
            target=command_server_loop, args=(command_queue, server_socket), daemon=True
        )
        command_thread.start()

        try:
            while not mic_stream.closed:
                data, _ = mic_stream.read(chunk_size)
                data = data.squeeze()

                original_audio_output.write(data)

                vad_result = vad_iterator(data)
                if vad_result is not None and "start" in vad_result:
                    log({"event": "speech_start", "start_frame": current_frame})
                    current_speech_segment = SpeechSegment(
                        start_frame=current_frame, chunks=[], transcriptions=[]
                    )

                    # add a bit of pre-padding of original audio
                    if len(prepad_buffer) > 0:
                        num_prepad_chunks = min(5, len(prepad_buffer))
                        current_speech_segment.chunks.extend(
                            list(prepad_buffer)[-num_prepad_chunks:]
                        )

                if current_speech_segment is not None:
                    current_speech_segment.chunks.append(data)
                else:
                    prepad_buffer.append(data)

                with input_queue_mutex:
                    while len(input_queue) > 0:
                        user_input = input_queue.popleft()
                        if user_input.strip() != "":
                            log(
                                {
                                    "event": "custom_input",
                                    "frame": current_frame,
                                    "content": user_input,
                                }
                            )
                            custom_segment = CustomInputSegment(
                                frame=current_frame, content=user_input
                            )

                            custom_input_segments.append(custom_segment)

                while len(command_queue) > 0:
                    command = command_queue.popleft()
                    log({"event": "command_received", "command": command.model_dump()})
                    if command.command == "insert_text":
                        text = command.text
                        if text.strip() != "":
                            log(
                                {
                                    "event": "custom_input",
                                    "frame": current_frame,
                                    "content": text,
                                }
                            )
                            custom_segment = CustomInputSegment(
                                frame=current_frame, content=text
                            )
                            custom_input_segments.append(custom_segment)
                    elif command.command == "screenshot":
                        log(
                            {
                                "event": "custom_input",
                                "frame": current_frame,
                                "content": command.path,
                            }
                        )
                        custom_segment = CustomInputSegment(
                            frame=current_frame, content=command.path
                        )
                        custom_input_segments.append(custom_segment)

                current_frame += data.shape[0]

                if (
                    current_speech_segment is not None
                    and vad_result is not None
                    and "end" in vad_result
                ):
                    log({"event": "speech_end", "end_frame": current_frame})

                    # TODO: remove non-speech from the end
                    # TODO: handle speech chunks longer than 30 seconds

                    # add some silence
                    current_speech_segment.chunks.extend(
                        [np.zeros(chunk_size) for _ in range(min_silence_chunks)]
                    )
                    speech_segments.append(current_speech_segment)
                    voice_audio_output.write(
                        np.concatenate(current_speech_segment.chunks)
                    )
                    voice_audio_current_frame += sum(
                        len(s) for s in current_speech_segment.chunks
                    )
                    current_speech_segment = None

                    while (
                        sum(len(s.chunks) for s in speech_segments)
                        > max_chunks_for_whisper
                    ):
                        segment = speech_segments.popleft()
                        drop_segment(segment)

                    chunk = np.concatenate(
                        [s for s in speech_segments for s in s.chunks]
                    )

                    inputs = processor(
                        chunk,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                        return_attention_mask=True,
                    ).to(device, dtype=torch_dtype)

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

                    generated_ids = result["sequences"]

                    tokens = generated_ids[0].tolist()
                    decoded = processor.tokenizer.convert_ids_to_tokens(tokens)

                    speech_segment_index = 0
                    speech_segment_start_timestamp = 0.0
                    text_so_far = []

                    for token_str, timestamp in zip(
                        decoded, result["token_timestamps"][0]
                    ):
                        # filter out special tokens `<|...|>`
                        if token_str.startswith("<|") and token_str.endswith("|>"):
                            continue

                        speech_segment = speech_segments[speech_segment_index]

                        if (
                            timestamp
                            > speech_segment_start_timestamp
                            + speech_segment.duration_seconds()
                            and speech_segment_index + 1 < len(speech_segments)
                        ):
                            speech_segment.transcriptions.append(
                                processor.tokenizer.convert_tokens_to_string(
                                    text_so_far
                                )
                            )
                            text_so_far = []
                            speech_segment_index += 1
                            speech_segment_start_timestamp = timestamp

                        text_so_far.append(token_str)

                    if len(text_so_far) > 0:
                        speech_segments[speech_segment_index].transcriptions.append(
                            processor.tokenizer.convert_tokens_to_string(text_so_far)
                        )

                    print("Segment buffer:")
                    print()
                    for s in speech_segments:
                        print(
                            f"Segment starting at {s.start_frame / SAMPLE_RATE:.2f}s, duration {s.duration_seconds():.2f}s:"
                        )
                        for t in s.transcriptions:
                            print(f" - {t}")

                    # decode to text
                    text = processor.batch_decode(
                        generated_ids, skip_special_tokens=True, language=language
                    )[0]
                    log({"event": "transcription", "text": text})

        except KeyboardInterrupt:
            print("\nExiting note taker. Goodbye!")

            while len(speech_segments) > 0:
                segment = speech_segments.popleft()
                drop_segment(segment)
            for cs in custom_input_segments:
                log(
                    {
                        "event": "custom_input_attached",
                        "frame": cs.frame,
                        "content": cs.content,
                    }
                )
                note_output.write(f"{cs.content}\n")


if __name__ == "__main__":
    fire.Fire(main)  # pyright: ignore[reportUnknownMemberType]
