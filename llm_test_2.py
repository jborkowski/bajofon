import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------
# CONFIG: model, device
# -----------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # example local LLM
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# EXAMPLE POLISH TRANSCRIPT
# -----------------------------
transcript = (
    "Dzień dobry wszystkim. Chciałbym rozpocząć spotkanie. "
    "Najpierw omówimy projekt A. Projekt A ma kilka wyzwań. "
    "Następnie przejdziemy do projektu B. Projekt B jest w fazie testów."
)

transcript = open("output.srt").read()

# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,  # reduce VRAM usage
    torch_dtype=torch.float16
)

# -----------------------------
# PROMPT LLM TO RETURN JSON INSTRUCTIONS
# -----------------------------
prompt = f"""
You are a formatting assistant. 
Analyze the Polish transcript below and output a JSON array of sentence number after which to add paragraph breaks.

Transcript:
{transcript}

Output JSON example:
[1,3,6]
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

llm_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("LLM output:\n", llm_output)

# -----------------------------
# PARSE JSON INSTRUCTIONS
# -----------------------------
try:
    instructions = json.loads(llm_output)
except json.JSONDecodeError:
    # fallback: try to extract JSON substring if LLM included extra text
    start = llm_output.find("[")
    end = llm_output.rfind("]") + 1
    instructions = json.loads(llm_output[start:end])

# -----------------------------
# APPLY INSTRUCTIONS
# -----------------------------
# Split transcript into sentences (simple split by period for demo)
sentences = [s.strip() for s in transcript.split(".") if s.strip()]
formatted_sentences = sentences.copy()

# Apply headings and paragraph breaks
output_lines = []
for i, sentence in enumerate(formatted_sentences, start=1):
    # Check for heading
    for instr in instructions:
        if instr["action"] == "heading" and instr.get("before_sentence") == i:
            output_lines.append(f"# {instr['text']}")
    
    # Add the sentence
    output_lines.append(sentence + ".")

    # Check for paragraph break
    for instr in instructions:
        if instr["action"] == "paragraph_break" and instr.get("after_sentence") == i:
            output_lines.append("")  # empty line = paragraph break

# Final formatted text
formatted_text = "\n".join(output_lines)
print("\nFormatted transcript:\n")
print(formatted_text)

