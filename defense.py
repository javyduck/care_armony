import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .architectures import robust_clip, load_gcn_from_ckpt
from typing import Optional
from art.estimators.classification import PyTorchClassifier

def get_care_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = CareModel(model_kwargs, wrapper_kwargs, weights_path)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
    
class CareModel(nn.Module):
    def __init__(self, model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None):
        super(CareModel, self).__init__()
        
        self.noise_sd = model_kwargs.get('noise_sd', 0.0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Set path
        self.checkpoint = torch.load(weights_path, map_location=self.device)
        dir_name, _ = os.path.split(weights_path)
        denoising_ckpt = os.path.join(dir_name, 'cifar10_uncond_50M_500K.pt')
        # Load GCN model from checkpoint
        self.gcn_model = load_gcn_from_ckpt(self.checkpoint, self.device)
        
        # Extract additional values from model_kwargs if needed
        # For example: some_value = model_kwargs.get('some_key', default_value)

        # Initialize base classifier
        self.base_classifier = robust_clip(self.checkpoint['clip_arch'],
                                                self.checkpoint['dataset'],
                                                reasoning=True,
                                                knowledge_path=self.checkpoint['knowledge_path'],
                                                noise_sd=self.noise_sd,
                                                denoising=True,
                                                denoising_ckpt = denoising_ckpt,
                                                gcn_model=self.gcn_model,
                                                use_classifier=self.checkpoint['classifier'],
                                                device = self.device)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Number of noisy samples per input
        num_samples = 100
        
        # Get the shape of the input tensor
        N = x.size(0)
        input_shape = x.size()[1:]
        
        if self.noise_sd == 0.:
            mean_outputs = self.base_classifier(x)
        else:
            # List to store mean outputs
            mean_outputs = []

            # Process each x[i] tensor separately
            for i in range(N):
                # Generate Gaussian noise for x[i]
                noise = torch.randn(num_samples, *input_shape).to(x.device) * self.noise_sd

                # Add noise to x[i]
                noisy_samples = x[i] + noise

                # Pass noisy samples through the model
                outputs = self.base_classifier(noisy_samples).mean(dim=0)

                mean_outputs.append(outputs)

            # Convert the list of mean outputs to a tensor
            mean_outputs = torch.stack(mean_outputs)
        
        return self.confidences_to_log_softmax(mean_outputs)

    def confidences_to_log_softmax(self, confidences: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        # Clamp the confidences to avoid log(0) and log(1)
        confidences = torch.clamp(confidences, epsilon, 1-epsilon)

        # Convert probabilities to logits
        logits = torch.log(confidences) - torch.log1p(-confidences)

        # Normalize logits for numerical stability
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values

        # Apply log_softmax
        log_softmax_values = F.log_softmax(logits, dim=-1)

        return log_softmax_values