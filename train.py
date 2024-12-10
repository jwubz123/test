import torch

import torch.nn as nn
import torch.optim as optim

# Define a simple model for demonstration purposes
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc_class = nn.Linear(16 * 32 * 32, num_classes)
        self.fc_bbox = nn.Linear(16 * 32 * 32, 4)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        class_logits = self.fc_class(x)
        bbox_preds = self.fc_bbox(x)
        return class_logits, bbox_preds

# Define the loss functions
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.MSELoss()

# Training pipeline
def train(dataloader, model, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, gt_classes, gt_bboxes in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, bbox_preds = model(images)
            
            # Calculate losses
            class_loss = class_loss_fn(class_logits, gt_classes.squeeze())
            bbox_loss = bbox_loss_fn(bbox_preds, gt_bboxes)
            total_loss = class_loss + bbox_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

# Example usage
if __name__ == "__main__":
    # Assuming dataloader is defined elsewhere
    num_classes = 10
    model = SimpleModel(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    
    # Replace `dataloader` with your actual dataloader
    train(dataloader, model, optimizer, num_epochs)