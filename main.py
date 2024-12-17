# This file conntains the latest code with the sample code in the
# use this file to implement generated using the LLM model if there
# are any problems or modification please do update accordingly
# the older logic is redudent and also deprecated

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class AdaptivePatchExtractor:
    def __init__(self, patch_sizes=[15, 17, 21], 
                 input_resolutions=[(640, 480), (1920, 1080)]):
        """
        Adaptive Patch Extractor for multiple camera resolutions
        
        Args:
            patch_sizes (list): Sizes of patches to extract
            input_resolutions (list): Supported camera input resolutions
        """
        self.patch_sizes = patch_sizes
        self.input_resolutions = input_resolutions
    
    def extract_patches(self, image):
        """
        Dynamically extract patches based on input image size
        
        Args:
            image (torch.Tensor): Input image tensor
        
        Returns:
            list: Extracted patches
        """
        _, _, height, width = image.size()
        patches = []
        
        for patch_size in self.patch_sizes:
            # Compute center coordinates
            start_h = (height - patch_size) // 2
            start_w = (width - patch_size) // 2
            
            # Extract center patch
            patch = image[:, :, 
                start_h:start_h+patch_size, 
                start_w:start_w+patch_size
            ]
            patches.append(patch)
        
        return patches

class ResolutionAdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 adaptive=True, base_width=64):
        """
        Adaptive Convolutional Layer
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Convolution kernel size
            adaptive (bool): Enable adaptive width scaling
            base_width (int): Base width for scaling
        """
        super(ResolutionAdaptiveConv, self).__init__()
        
        self.adaptive = adaptive
        
        # Base convolution
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        # Adaptive width scaling
        if adaptive:
            self.width_scaler = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.ReLU()
            )
    
    def forward(self, x):
        """
        Forward pass with optional adaptive scaling
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Processed tensor
        """
        x = self.conv(x)
        
        if self.adaptive:
            # Adaptive width scaling based on input resolution
            scaling = self.width_scaler(x)
            x = x * scaling
        
        return x

class ResolutionAdaptiveAutoencoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        """
        Resolution-adaptive Autoencoder
        
        Args:
            input_channels (int): Number of input channels
            base_channels (int): Base number of channels
        """
        super(ResolutionAdaptiveAutoencoder, self).__init__()
        
        # Encoder with adaptive convolutions
        self.encoder = nn.Sequential(
            ResolutionAdaptiveConv(input_channels, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            ResolutionAdaptiveConv(base_channels, base_channels*2),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            ResolutionAdaptiveConv(base_channels*2, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU()
        )
        
        # Decoder with adaptive convolutions
        self.decoder = nn.Sequential(
            ResolutionAdaptiveConv(base_channels*4, base_channels*2),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            ResolutionAdaptiveConv(base_channels*2, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            ResolutionAdaptiveConv(base_channels, input_channels),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Reconstructed input
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CameraInputAnomalyDetector(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 patch_sizes=[15, 17, 21],
                 input_resolutions=[(640, 480), (1920, 1080)],
                 anomaly_threshold=0.05):
        """
        Comprehensive Anomaly Detector for Camera Inputs
        
        Args:
            input_channels (int): Number of input channels
            patch_sizes (list): Sizes of patches to extract
            input_resolutions (list): Supported camera input resolutions
            anomaly_threshold (float): Threshold for anomaly detection
        """
        super(CameraInputAnomalyDetector, self).__init__()
        
        # Patch extractor
        self.patch_extractor = AdaptivePatchExtractor(
            patch_sizes, input_resolutions
        )
        
        # Preprocessing transforms
        self.preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),  # Standardize input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Autoencoder branches
        self.autoencoder_branches = nn.ModuleList([
            ResolutionAdaptiveAutoencoder(input_channels) 
            for _ in patch_sizes
        ])
        
        # Anomaly parameters
        self.patch_sizes = patch_sizes
        self.anomaly_threshold = anomaly_threshold
    
    def preprocess_input(self, image):
        """
        Preprocess input image for consistent processing
        
        Args:
            image (torch.Tensor or numpy.ndarray): Input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to tensor if numpy array
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Preprocess
        return self.preprocessor(image).unsqueeze(0)
    
    def detect_anomalies(self, image):
        """
        Detect anomalies in camera input
        
        Args:
            image (torch.Tensor or numpy.ndarray): Input image
        
        Returns:
            dict: Anomaly detection results
        """
        # Preprocess input
        processed_image = self.preprocess_input(image)
        
        # Extract patches
        patches = self.patch_extractor.extract_patches(processed_image)
        
        # Anomaly results
        anomaly_results = {
            'is_anomalous': False,
            'anomalous_patches': []
        }
        
        # Process each patch
        for idx, (patch_size, patch, autoencoder) in enumerate(
            zip(self.patch_sizes, patches, self.autoencoder_branches)
        ):
            # Reconstruct patch
            reconstructed = autoencoder(patch)
            
            # Compute reconstruction error
            recon_error = F.mse_loss(patch, reconstructed, reduction='none')
            mean_error = recon_error.mean()
            
            # Check for anomaly
            if mean_error > self.anomaly_threshold:
                anomaly_results['is_anomalous'] = True
                anomaly_results['anomalous_patches'].append({
                    'patch_id': idx,
                    'patch_size': patch_size,
                    'reconstruction_error': mean_error.item()
                })
        
        return anomaly_results
    
    def forward(self, image):
        """
        Forward pass for training or inference
        
        Args:
            image (torch.Tensor or numpy.ndarray): Input image
        
        Returns:
            dict: Anomaly detection results
        """
        return self.detect_anomalies(image)

# Example usage and testing
def main():
    # Create anomaly detector
    anomaly_detector = CameraInputAnomalyDetector(
        input_channels=3,
        patch_sizes=[15, 17, 21],
        input_resolutions=[(640, 480), (1920, 1080)],
        anomaly_threshold=0.05
    )
    
    # Simulate camera inputs (VGA and HD)
    vga_input = torch.randn(3, 480, 640)  # VGA camera input
    hd_input = torch.randn(3, 1080, 1920)  # HD camera input
    
    # Detect anomalies
    vga_anomalies = anomaly_detector(vga_input)
    hd_anomalies = anomaly_detector(hd_input)
    
    # Print results
    print("VGA Camera Anomaly Detection:")
    print(f"Is Anomalous: {vga_anomalies['is_anomalous']}")
    print("HD Camera Anomaly Detection:")
    print(f"Is Anomalous: {hd_anomalies['is_anomalous']}")

if __name__ == "__main__":
    main()
