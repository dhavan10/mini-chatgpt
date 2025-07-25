from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the trained model and tokenizer
model_path = "./mini-chatgpt-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Mini ChatGPT is ready to chat! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip user input from beginning of response (optional, depends on your dataset)
    cleaned_response = response.replace(user_input, "").strip()

    print(f"Bot: {cleaned_response}\n")
