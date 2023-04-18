import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.utils.prune as prune
import numpy as np
import os
from models.resnet import BasicBlock, Bottleneck, ResNet56 as resnet56
import dill  # in order to save Lambda Layer
import matplotlib.pyplot as plt
import seaborn as sns

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_ids = [0, 1]

# Load model checkpoint
model_path = os.path.expanduser('~/pytorch_resnet_cifar10/pytorch-cifar/checkpoint/ckpt.pth')
checkpoint = torch.load(model_path)
net = resnet56()
net = torch.nn.DataParallel(net)
net.load_state_dict(checkpoint['net'])
torch.save(net.module, 'resnet56_check_point.pth', pickle_module=dill)

# Load the converted pretrained model
model = torch.load('resnet56_check_point.pth')
model = model.to(device)

#define evaluation function
def evaluate_model_on_cifar10(model, device):
    # set up CIFAR10 dataset and data loader
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # set up criterion and counter
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    # evaluate model on test set
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()    
    accuracy = 100 * correct / total
    return accuracy

accuracy_before_pruning = evaluate_model_on_cifar10(model, device)
print(f"Accuracy before pruning: {accuracy_before_pruning:.2f}%")


# Get the BN layers in the desired blocks
bn_layers = []
for m in model.modules():
    if isinstance(m, nn.Sequential):
        for n in m:
            if isinstance(n, nn.Sequential) and hasattr(n[0], 'conv1x1'):
                bn1_layer = None
                bn3_layer = None
                for o in n:
                    if isinstance(o, nn.BatchNorm2d) and hasattr(o, 'weight') and hasattr(o, 'bias'):
                        if bn1_layer is None:
                            bn1_layer = o
                        else:
                            bn3_layer = o
                            bn_layers.append((bn1_layer, bn3_layer))
                            break

# Define the pruning parameters
pruning_perc = 0.1  # Percentage of channels to prune
pruning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]

num_masks = 100  # Number of random masks to use

# Create a list to store the accuracies
accuracies_random_masks  = []

# Prune the model using random masks and record the accuracies
for pruning_perc in pruning_percentages:  # Add this line
    for i in range(num_masks):
        # Generate random binary masks for the BN layers in each basic block
        masks = []
        for block in model.modules():
            if isinstance(block, BasicBlock):
                # Get the BN layers of the BasicBlock
                bn_layers = [block.bn1, block.bn2]

                # Generate random binary masks for the BN layers
                for layer in bn_layers:
                    mask = torch.zeros_like(layer.weight)
                    mask.view(-1)[torch.randperm(mask.numel())[:int(pruning_perc * mask.numel())]] = 1
                    masks.append(mask)

    # Prune the BN layers using the masks
    mask_index = 0
    for block in model.modules():
        if isinstance(block, BasicBlock):
            bn_layers = [block.bn1, block.bn2]
            for j, layer in enumerate(bn_layers):
                prune.custom_from_mask(layer, name='weight', mask=masks[mask_index])
                mask_index += 1

    # Evaluate the pruned model on CIFAR10
    accuracy = evaluate_model_on_cifar10(model, device)

    # Record the accuracy
    accuracies_random_masks .append(accuracy)

# Experimenting with magnitude pruning
def magnitude_prune(layer, pruning_perc):
    # Calculate the threshold value based on pruning percentage
    threshold = torch.topk(torch.abs(layer.weight.view(-1)), int(pruning_perc * layer.weight.numel()), largest=False)[0][-1]
    # Create a mask based on the threshold
    mask = torch.where(torch.abs(layer.weight) > threshold, torch.ones_like(layer.weight), torch.zeros_like(layer.weight))
    return mask

for pruning_perc in pruning_percentages:
    # Apply magnitude-based pruning and record accuracy
    masks_magnitude_pruning = [magnitude_prune(layer, pruning_perc) for layer in bn_layers]

    for j, layer in enumerate(bn_layers):
        prune.custom_from_mask(layer, name='weight', mask=masks_magnitude_pruning[j])
    
    accuracy_magnitude_pruning = evaluate_model_on_cifar10(model, device)
    print(f"Accuracy after Magnitude pruning with {pruning_perc*100:.0f}% pruning: {accuracy:.2f}%")

model = torch.load('resnet56_check_point.pth')
model = model.to(device)

# Experimenting with magnitude pruning
def l1_prune(layer, pruning_perc):
    l1_norm = layer.weight.abs()
    num_prune = int(pruning_perc * layer.weight.size(0))
    threshold = torch.topk(l1_norm.view(layer.weight.size(0), -1).sum(dim=1), num_prune, largest=False)[0][-1]
    mask = torch.where(l1_norm > threshold, torch.ones_like(layer.weight), torch.zeros_like(layer.weight))
    return mask

masks_l1_pruning = [l1_prune(layer, pruning_perc) for layer in bn_layers]

# Prune the BN layers using the masks
for j, layer in enumerate(bn_layers):
    prune.custom_from_mask(layer, name='weight', mask=masks_l1_pruning[j])

accuracy_l1_pruning = evaluate_model_on_cifar10(model, device)
print(f"Accuracy after L1 Norm pruning: {accuracy_l1_pruning:.2f}%")

# Experimenting with Group Lasso pruning
def group_lasso_prune(layer, pruning_perc):
    weight_groups = layer.weight.view(layer.weight.shape[0], -1)
    group_norms = weight_groups.norm(p=2, dim=1)
    num_prune = int(pruning_perc * len(group_norms))
    _, prune_indices = torch.topk(group_norms, num_prune, largest=False)
    mask = torch.ones_like(layer.weight)
    mask[prune_indices] = 0
    return mask

# Get the convolutional layers in the desired blocks
conv_layers = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        conv_layers.append(m)


# Apply Group Lasso pruning and compute accuracy
masks_group_lasso = [group_lasso_prune(layer, pruning_perc) for layer in bn_layers]

# Function to apply masks to layers
def apply_masks(layers, masks):
    for i, layer in enumerate(layers):
        prune.custom_from_mask(layer, name='weight', mask=masks[i])

apply_masks(bn_layers, masks_group_lasso)
accuracy_group_lasso = evaluate_model_on_cifar10(model, device)

# Print the accuracy for Group Lasso pruning
print("Group Lasso pruning accuracy: {:.4f}".format(accuracy_group_lasso))



# Plot the accuracy distribution for random masks
sns.histplot(accuracies_random_masks, kde=True, color='blue', label='Random Masks')

# Plot the single accuracy value for magnitude-based pruning
#plt.axvline(x=accuracy_l1_pruning, color='red', linestyle='--', label='L1-Norm Pruning')
#plt.axvline(x=accuracy_group_lasso, color='green', linestyle='--', label='Group Lasso Pruning')
plt.axvline(x=accuracy_magnitude_pruning, color='yellow', linestyle='--', label='Magnitude Pruning')


plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Accuracy Distribution Comparison')
plt.legend()
plt.savefig("L1_Lasso_magnitude_accuracy_comparison.png")


# Calculate the mean accuracy and standard deviation of the random masks
mean_acc_random_masks = sum(accuracies_random_masks) / len(accuracies_random_masks)
std_acc_random_masks = torch.tensor(accuracies_random_masks).std()

# Calculate the Z-score for the magnitude-based pruning accuracy
z_score = (accuracy_magnitude_pruning - mean_acc_random_masks) / std_acc_random_masks

print("Z-score of magnitude-based pruning: {:.4f}".format(z_score))

# Analyze the results based on the Z-score
if z_score < 1:
    print("Z-score is small. Consider finding better pruning masks.")
elif 1 < z_score < 2:
    print("Z-score is moderate. It would take dozens of random tests to do better.")
elif z_score > 3:
    print("Z-score is large. Consider applying LR-rewinding or fine-tuning to improve the results.")