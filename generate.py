from transformers import pipeline

print("Loading your fine-tuned GPT-2 model...")

generator = pipeline(
    "text-generation",
    model="./model",
    tokenizer="./model"
)

prompt = input("Enter a prompt: ")

result = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True
)

print("\nGenerated Text:\n")
print(result[0]["generated_text"])