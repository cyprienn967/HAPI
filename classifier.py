import torch
import torch.nn as nn

class HAPIClassifier(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super(HAPIClassifier, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Define classifier architecture
        self.model = nn.Sequential(
            nn.Linear(4096 * 2, 256),  # First linear layer
            nn.ReLU(),
            nn.Linear(256, 128),       # Second linear layer
            nn.ReLU(),
            nn.Linear(128, 64),        # Third linear layer
            nn.ReLU(),
            nn.Linear(64, 2)           # Output layer (binary classification)
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]

        # Fix the mismatch in saved state dict key names
        corrected_state_dict = {
            "0.weight": state_dict["linear1.weight"],
            "0.bias": state_dict["linear1.bias"],
            "2.weight": state_dict["linear2.weight"],
            "2.bias": state_dict["linear2.bias"],
            "4.weight": state_dict["linear3.weight"],
            "4.bias": state_dict["linear3.bias"],
            "6.weight": state_dict["linear4.weight"],
            "6.bias": state_dict["linear4.bias"]
        }

        self.model.load_state_dict(corrected_state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def classify(self, token_tensor):
        """
        Returns True if the token is classified as hallucination, otherwise False.
        The input tensor is cast to float32 to match the classifier's weights.
        """
        token_tensor = token_tensor.to(self.device).float()  # Ensure matching dtype
        with torch.no_grad():
            output = self.model(token_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction == 1  # 1 means hallucination, 0 means valid

    def get_hallucination_score(self, token_tensor):
        """
        Returns the probability (between 0 and 1) that the token is hallucinated.
        Lower scores are better (more likely valid).
        """
        token_tensor = token_tensor.to(self.device).float()
        with torch.no_grad():
            output = self.model(token_tensor)
            probs = torch.softmax(output, dim=1)
            # index 1 corresponds to hallucination probability
            score = probs[0, 1].item()
        return score
