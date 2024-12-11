import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MultiPatch(nn.Module):
    def __init__(self, input_channels=3, patch_sizes=[15, 21, 25], num_classes=10):
        super(MultiPatchModel, self).__init__()
        
        # Patch extraction layers
        self.patch_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Unfold(kernel_size=patch_size, stride=patch_size),
                nn.Flatten(start_dim=1)
            ) for patch_size in patch_sizes
        ])
        
        # Calculate the total feature dimension after patch extraction
        def calculate_patch_features(image_size, patch_sizes):
            patch_features = []
            for patch_size in patch_sizes:
                patches_per_dim = image_size // patch_size
                patch_features.append(patches_per_dim * patches_per_dim)
            return patch_features
        
        # Assume a standard input image size (e.g., 224x224)
        image_size = 224
        patch_feature_counts = calculate_patch_features(image_size, patch_sizes)
        total_patch_features = sum(
            count * (patch_size * patch_size * input_channels) 
            for count, patch_size in zip(patch_feature_counts, patch_sizes)
        )
        
        # MLP for merging patches
        self.merger = nn.Sequential(
            nn.Linear(total_patch_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract patches from different sizes
        patch_features = [
            extractor(x) for extractor in self.patch_extractors
        ]
        
        # Concatenate all patch features
        merged_patches = torch.cat(patch_features, dim=1)
        
        # Pass through MLP
        output = self.merger(merged_patches)
        
        return output

(* # Example usage
def main():
    # Create model instance
    model = MultiPatchModel(
        input_channels=3,  # RGB images
        patch_sizes=[15, 21, 25],
        num_classes=10  # e.g., for CIFAR-10
    )
    
    # Example input (batch_size, channels, height, width)
    x = torch.randn(32, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main() *)
