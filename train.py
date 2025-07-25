from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

texts = [
    "Hello, how are you?",
    "I am a mini chat model.",
    "This is a test sentence.",
    "Let's train a transformer.",
    "Transformers are powerful!"
]

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
# Add labels for training
encodings["labels"] = encodings["input_ids"].clone()

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings.input_ids.size(0)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = DummyDataset(encodings)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10,
    evaluation_strategy="no",
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./mini-chatgpt-model")
tokenizer.save_pretrained("./mini-chatgpt-model")
