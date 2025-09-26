# Bajofon

Audio processing and note-taking application with AI transcription.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installation

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   # Install core dependencies
   uv sync --no-install-project
   
   # Install with optional LLM dependencies
   uv sync --no-install-project --extra llm
   ```

### Running the Project

Use `uv run` to execute Python scripts with the project environment:

```bash
# Run the main note taker
uv run python note_taker.py

# Run LLM tests
uv run python llm_test.py
uv run python llm_test_2.py
```

### Dependencies

**Core dependencies:**
- transformers - Hugging Face transformers library
- torch - PyTorch for machine learning
- sounddevice - Audio recording/playback
- soundfile - Audio file I/O
- scipy - Scientific computing
- numpy - Numerical computing
- silero-vad - Voice activity detection
- fire - Command-line interface generation

**Optional LLM dependencies (`--extra llm`):**
- accelerate - Hugging Face accelerate for faster training/inference
- bitsandbytes - Memory efficient optimizers (Linux only)
- sentencepiece - Tokenization library