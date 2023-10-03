import sys
sys.path.append('..')
import torch
import torch.nn as nn
from architectures import robust_clip, load_gcn_from_ckpt


class CareModel(nn.Module):
    def __init__(self, noise_sd=0.0, path='pretrained_models/gcn_ckpt/checkpoint.pth.tar'):
        super(CareModel, self).__init__()
        
        self.noise_sd = noise_sd
        
        # Load checkpoint
        self.checkpoint = torch.load(path)
        
        # Load GCN model from checkpoint
        self.gcn_model = load_gcn_from_ckpt(self.checkpoint)
        
        # Initialize base classifier
        self.base_classifier = robust_clip(self.checkpoint['clip_arch'],
                                                self.checkpoint['dataset'],
                                                reasoning=True,
                                                knowledge_path=self.checkpoint['knowledge_path'],
                                                noise_sd=noise_sd,
                                                denoising=True,
                                                gcn_model=self.gcn_model,
                                                use_classifier=self.checkpoint['classifier'])

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Number of noisy samples per input
        num_samples = 100
        
        # Get the shape of the input tensor
        N = x.size(0)
        input_shape = x.size()[1:]
        
        if self.noise_sd == 0.:
            
            return self.base_classifier(x)
        
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
        
            return mean_outputs

