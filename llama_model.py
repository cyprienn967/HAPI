import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class LlamaChat7B:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=400):
        """
        Generates text using the Llama-2 model.
        The max_length has been increased to produce roughly 200-300 words.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_token_by_token(self, prompt, classifier, max_length=400, temperature=1.0, top_k=50, max_attempts=5):
        """
        Generates text one token at a time with hallucination filtering.
        For each token, a candidate is sampled using temperature-scaled top-k sampling.
        The candidate token’s hidden state (from the last two layers) is checked by the classifier.
        If it is flagged as a hallucination, re-sampling occurs (up to max_attempts) before accepting a token.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.clone()
        hallucinations = []

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(generated_ids, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
                logits = logits / temperature
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_values, dim=-1)

            valid_token_found = False
            chosen_token_id = None

            for attempt in range(max_attempts):
                sampled_index = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)
                sampled_index_scalar = sampled_index.item()
                token_id = top_k_indices[0, sampled_index_scalar].unsqueeze(0)  # shape: (1,)

                candidate_input_ids = torch.cat([generated_ids, token_id.unsqueeze(0)], dim=1)
                with torch.no_grad():
                    candidate_outputs = self.model(candidate_input_ids, output_hidden_states=True)
                    candidate_hidden_state = torch.cat(
                        (candidate_outputs.hidden_states[-1][:, -1, :],
                         candidate_outputs.hidden_states[-2][:, -1, :]),
                        dim=1
                    )
                if classifier.classify(candidate_hidden_state):
                    hallucinations.append(self.tokenizer.decode(token_id))
                    continue
                else:
                    chosen_token_id = token_id
                    valid_token_found = True
                    break

            if not valid_token_found:
                chosen_token_id = top_k_indices[0, 0].unsqueeze(0)

            generated_ids = torch.cat([generated_ids, chosen_token_id.unsqueeze(0)], dim=1)

            if chosen_token_id.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text, hallucinations
