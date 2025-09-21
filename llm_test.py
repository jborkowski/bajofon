from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device=device,
    load_in_4bit=True,
    torch_dtype=torch.float16
)

print("model loaded")

prompt = """Rewrite the following audio transcript into a well-formatted document with paragraphs, headings, and punctuation.
DO NOT make up any new text in paragraphs, only change formatting. But you can invent heading names. Retain original language, DO NOT TRANSLATE.

Original transcript:

    """ + open("output.srt").read() + "\n\nRewritten transcript:\n\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False  # greedy decoding for deterministic output
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)

