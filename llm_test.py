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
    device_map='auto',
    load_in_4bit=True,
    torch_dtype=torch.float16
)

print("model loaded")

messages = [
    {"role": "system", "content": """
You are an expert editor. Rewrite the following Polish transcript into a clean, readable document:
- Keep all text in Polish
- Add proper punctuation and capitalization
- Remove fillers like "Hmmm" and "Ummm"
- Insert paragraphs and headings if appropriate (Markdown syntax, heading names in Polish)
     """,},
    {"role": "user", "content": open("output.srt").read()},
 ]

messages = [
    {"role": "system", "content": """
     Jesteś ekspertem w edycji tekstu. Przekształć poniższy zapis rozmowy w czytelny dokument:
- Zachowaj język polski
- Dodaj interpunkcję i wielkie litery
- Wstaw akapity i nagłówki tam, gdzie to stosowne
     """,},
    {"role": "user", "content": open("output.srt").read()},
 ]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

print(tokenizer.decode(inputs[0]))

with torch.no_grad():
    output_ids = model.generate(
        inputs,
        max_new_tokens=500,
        do_sample=False  # greedy decoding for deterministic output
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)

