from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Load dataset
with open("dataset/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = Dataset.from_dict({"text": [text]})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training settings
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete. Model saved in ./model")