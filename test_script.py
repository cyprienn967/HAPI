#!/usr/bin/env python
import sys
from llama_model import LlamaChat7B

def main():
    # Initialize the LlamaChat7B model (will use GPU if available)
    model = LlamaChat7B()
    
    # Define a list of example prompts to test the model.
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
        "Explain the theory of relativity in simple terms."
    ]
    
    for prompt in prompts:
        print("Prompt:", prompt)
        response = model.generate_text(prompt, max_length=400)
        print("Response:", response)
        print("-" * 80)

if __name__ == '__main__':
    main()
