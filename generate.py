import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from classifier import HAPIClassifier  # Import hallucination classifier
import os
from huggingface_hub import login

# Authenticate to Hugging Face
login(token="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")

# Set model paths
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
CLASSIFIER_PATH = "./models/best_acc_model.pt"  # Ensure classifier is in models/
PROMPTS_FILE = "./prompts.txt"

# Output file paths
OUTPUT_DIR = "./outputs"
BASELINE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_output.txt")
FILTERED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "filtered_output.txt")
HALLUCINATIONS_LOG_FILE = os.path.join(OUTPUT_DIR, "flagged_hallucinations.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Llama-2 model and tokenizer
print(f"Loading Llama-2 model on {device}...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

# Load hallucination classifier
print("Loading Hallucination Classifier...")
classifier = HAPIClassifier(CLASSIFIER_PATH, device=device)

# Read prompts from file (each line is one prompt)
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# Store results
baseline_results = []
filtered_results = []
flagged_hallucinations = []

def extract_hidden_states(input_ids):
    """
    Extracts and concatenates the representations of the last token from the last two hidden layers.
    This is used to form the input to the hallucination classifier.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_layer = hidden_states[-1][:, -1, :]         # (batch_size, hidden_dim)
        second_last_layer = hidden_states[-2][:, -1, :]    # (batch_size, hidden_dim)
        concatenated = torch.cat((last_layer, second_last_layer), dim=1)
    return concatenated

def generate_sentence_candidates(context, sentence_max_length=60, num_candidates=3):
    """
    Generates multiple candidate sentences using the provided context.
    """
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=sentence_max_length,
        do_sample=True,
        top_k=50,
        num_return_sequences=num_candidates,
    )
    candidates = []
    for i in range(num_candidates):
        sentence = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        # Remove the context from the generated text if accidentally included
        if sentence.startswith(context):
            sentence = sentence[len(context):].strip()
        # Split into sentences and take the first complete sentence
        if ". " in sentence:
            sentence = sentence.split(". ")[0] + "."
        else:
            if sentence and sentence[-1] not in ".!?":
                sentence += "."
        candidates.append(sentence.strip())
    return candidates

def compute_sentence_hallucination_score(sentence, classifier, original_prompt):
    """
    Computes an average hallucination score for a sentence by token.
    Lower scores mean less likely to be hallucinated.
    """
    # Remove the original prompt and newlines from the sentence.
    stripped_sentence = sentence.replace(original_prompt, "").replace("\n", " ").strip()
    tokens = stripped_sentence.split()
    if not tokens:
        return 0.0  # No tokens means no hallucination.
    total_score = 0.0
    for token in tokens:
        token_tensor = tokenizer(token, return_tensors="pt").input_ids.to(model.device)
        hidden_state = extract_hidden_states(token_tensor)
        score = classifier.get_hallucination_score(hidden_state)
        total_score += score
    avg_score = total_score / len(tokens)
    return avg_score

def generate_with_hallucination_filtering(prompt, classifier, desired_word_count=250, sentence_max_length=60, max_regen_attempts=5, num_candidates=3, threshold=0.5):
    """
    Generates a passage sentence-by-sentence with real-time hallucination filtering.
    For each sentence, multiple candidate sentences are generated.
    The candidate with the lowest hallucination score (below threshold) is chosen.
    If no candidate meets the threshold within max_regen_attempts, the best candidate is used.
    """
    context = prompt
    generated_sentences = []
    hallucinated_sentences = []  # For logging purposes

    while len(" ".join(generated_sentences).split()) < desired_word_count:
        attempts = 0
        selected_sentence = None
        best_score = float('inf')
        best_candidate = None

        while attempts < max_regen_attempts:
            candidates = generate_sentence_candidates(context, sentence_max_length, num_candidates)
            for sentence in candidates:
                score = compute_sentence_hallucination_score(sentence, classifier, prompt)
                # Save best candidate seen so far.
                if score < best_score:
                    best_score = score
                    best_candidate = sentence
            # If the best candidate is acceptable, choose it.
            if best_score < threshold:
                selected_sentence = best_candidate
                break
            else:
                print(f"🔴 Attempt {attempts+1}: Best candidate score {best_score:.3f} not below threshold {threshold}. Regenerating...")
                hallucinated_sentences.append(best_candidate)
                attempts += 1

        # If after max attempts no candidate is below threshold, use the best candidate.
        if selected_sentence is None:
            selected_sentence = best_candidate
        generated_sentences.append(selected_sentence)
        context += " " + selected_sentence

        # Safety check: if generation stalls (empty sentence), break.
        if not selected_sentence.strip():
            break

    generated_text = " ".join(generated_sentences)
    return generated_text, hallucinated_sentences

def generate_full_text(prompt, max_length=400):
    """
    Generates a full passage without sentence-level filtering.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Process each prompt: generate a baseline response and a filtered (hallucination-checked) response.
for i, prompt in enumerate(prompts):
    try:
        print(f"({i+1}/{len(prompts)}) Generating for prompt: {prompt}")

        # Baseline generation without hallucination filtering.
        baseline_output = generate_full_text(prompt, max_length=400)

        # Generation with real-time hallucination detection (sentence-by-sentence regeneration with candidate re-ranking).
        filtered_output, hallucinations = generate_with_hallucination_filtering(
            prompt, classifier, desired_word_count=250, sentence_max_length=60, max_regen_attempts=5, num_candidates=3, threshold=0.5
        )

        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT:\n{baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT:\n{filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(
                f"PROMPT: {prompt}\nFLAGGED SENTENCES: {', '.join(hallucinations)}\n"
            )

    except Exception as e:
        print(f"⚠️ Error processing prompt: {prompt}\n{e}")

# Save outputs to disk
with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(baseline_results))

with open(FILTERED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered_results))

with open(HALLUCINATIONS_LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(flagged_hallucinations))

print("\n✅ Generation complete! Check the outputs directory for results.")
