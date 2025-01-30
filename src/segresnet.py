import torch
from monai.networks.nets import SegResNet

# Define the SegResNet parameters
model = SegResNet(     
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=3,
    dropout_prob=0.2, 
)

# Set the model to evaluation mode
model.eval()

# Example input tensor (Batch Size, Channels, Depth, Height, Width)
input_tensor = torch.randn(1, 36, 384, 384)  

# Perform a forward pass
with torch.no_grad():  # Disable gradient computation
    output_tensor = model(input_tensor)

# Print the input and output shapes
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")