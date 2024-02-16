# %%
# Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch_lr_finder import LRFinder
from torchvision.utils import save_image
import wandb
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import pandas as pd
from dotenv import dotenv_values
import os
from datetime import datetime
from tqdm.notebook import tqdm
from collections import defaultdict

# %%
# Run utility bools
wandb_enabled = True
wandb_grad_tracking = False
save_best_model = False
save_incorrect_images_local = True


# %%
# Seed freeze
torch.manual_seed(1220)

# %%
# Parameters
param_config={'test_split': 0.1, # proportion of data is assigned to validation
              'val_split': 0.1, # proportion of data assigned to testing
              'img_size': (224, 224), # WxH of image to transform to
              'complex_rand_image_transform_enabled': True, # Whether to apply more complex random image transformations such as rand_flips
              'batch_size':32,
              'starting_lr': 1e-7, # Should be 1e-7 if using lr_finder
              'lr_finder_bool': True, # Wether
              'epochs': 5
        }

# wandb init
if wandb_enabled:
    wandb_enabled_str = 'online'
else:
    wandb_enabled_str = 'disabled'
    
os.environ["WANDB_API_KEY"] = dotenv_values(".env")['WANDB_API_KEY']
wandb.login()
wandb.init(project="climb_classifier_rear_glory_topo_custom_cnn",
            mode=wandb_enabled_str,
            config=param_config
            )

# Store into vars for readability
test_split = wandb.config['test_split']
val_split = wandb.config['val_split'] 
img_size = wandb.config['img_size']
complex_rand_image_transform_enabled = wandb.config['complex_rand_image_transform_enabled']
batch_size = wandb.config['batch_size']
starting_lr = wandb.config['starting_lr']
lr_finder_bool = wandb.config['lr_finder_bool']
epochs = wandb.config['epochs']

# %%
# Training device selection
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
# Image transformation
if complex_rand_image_transform_enabled:
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
else:
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

# %%
# Data loading
data_path = './climbing_classifier_data'
full_dataset = datasets.ImageFolder(root=data_path, transform=img_transform)

# Class label counts
class_labels = full_dataset.classes
num_classes = len(class_labels)
class_labels_dict = {i: class_labels[i] for i in range(num_classes)}
class_indices = {label: [] for label in class_labels}

for idx, (_, label_idx) in enumerate(full_dataset.imgs):
    class_name = class_labels[label_idx]
    class_indices[class_name].append(idx)

class_label_counts = {label: len(indices) for label, indices in class_indices.items()}

for label, count in class_label_counts.items():
    print(f"{label}: {count} samples\n")

# Data splitting
train_data, val_data, test_data = random_split(full_dataset, [1-(val_split+test_split), val_split, test_split])

# Data loader setup
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Data size confirmation
for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
num_img_channels = X.shape[1]
final_img_height = X.shape[2]
final_img_width = X.shape[3]


# %%
# Model definition
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=num_img_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten())
        
        n_channels = self.feature_extractor(torch.empty(X.shape)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_channels, out_features=600),
            nn.Dropout(0.25),
            nn.Linear(in_features=600, out_features=120),
            nn.Linear(in_features=120, out_features=num_classes) # Softmax is applied by the crossentropyloss function, so is not needed here.
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)        
        return out

model = CNN().to(device)
print(model)

# %%
# Loss function and optimizer definition
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)

# [Optional]
# LR Finder
if lr_finder_bool:
    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    lr_finder.range_test(train_dataloader, end_lr=1, num_iter=100, step_mode='exp')
    lr_finder.plot() # to inspect the loss-learning rate graph
    def suggested_lr(lr_finder, starting_lr):
        lrs = np.array(lr_finder.history["lr"])
        losses = np.array(lr_finder.history["loss"])

        min_grad_idx = None
        try:
            min_grad_idx = (np.gradient(np.array(losses))).argmin()
            output_lr = lrs[min_grad_idx]
        except ValueError:
            print("Failed to compute the gradients, there might not be enough points.")
            output_lr = starting_lr
            
        return output_lr
    optimized_lr = suggested_lr(lr_finder, starting_lr)


    lr_finder.reset() # to reset the model and optimizer to their initial state

    # Redefine optimizer with found LR
    optimizer = torch.optim.Adam(model.parameters(), lr=optimized_lr)

# %%
# Training loop definition
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        
        if batch % 3 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>4f}  [{current:>5d}/{size:>5d}]")
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(dataloader)

    # Log average loss to wandb
    wandb.log({"Training Loss": avg_loss}, step=epoch)
        

# %%
# Validation definition
best_accuracy = 0.0
best_model_weights = None
best_epoch = None
def val(dataloader, model, loss_fn, epoch):
    global best_accuracy
    global best_model_weights
    global best_epoch
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>4f} \n")
    
    wandb.log({"Val Loss": val_loss, "Val Accuracy": 100 * correct}, step=epoch)
        
    # Check if the current model is the best so far
    if correct > best_accuracy:
        best_accuracy = correct
        best_model_weights = model.state_dict()
        best_epoch = epoch+1
        print(f'New Best Model: Epoch:{best_epoch} | Accuracy:{best_accuracy*100:>4f}\n')

# %%
# Run Train/Val
if wandb_grad_tracking:
    wandb.watch(model, loss_fn, log='gradients', log_freq=25)
    
for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
    print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, epoch)
    val(val_dataloader, model, loss_fn, epoch)
    
print("Done!")
# %%
# Save the best model
if save_best_model:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = type(model).__name__
    filename = f"{model_name}_{timestamp}.pth"
    torch.save(best_model_weights, filename)
    print(f'Epoch {best_epoch} saved as {filename}')
# %%
# Test Data
# Utility funcs for test data output
def calculate_metrics(predictions, targets, num_classes):
    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Flatten the arrays to 1D
    predictions_flat = predictions_np.argmax(axis=1).flatten()
    targets_flat = targets_np.flatten()

    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(targets_flat, predictions_flat, labels=range(num_classes), average=None)

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

    return metrics_df

def save_incorrect_images(predictions, targets, images, class_labels_dict, base_dir='incorrect_image_archive'):
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the timestamped directory
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    incorrect_indices = torch.where(predictions.argmax(1) != targets)[0]

    for index in incorrect_indices:
        predicted_class = predictions[index].argmax().item()
        target_class = targets[index].item()

        # Get class labels using the dictionary
        predicted_class_label = class_labels_dict.get(predicted_class, f"Class {predicted_class}")
        target_class_label = class_labels_dict.get(target_class, f"Class {target_class}")

        # Create subfolders for true and predicted classes
        true_class_folder = os.path.join(save_dir, target_class_label)
        predicted_class_folder = os.path.join(true_class_folder, predicted_class_label)

        # Create subfolders if they don't exist
        os.makedirs(true_class_folder, exist_ok=True)
        os.makedirs(predicted_class_folder, exist_ok=True)

        # Save the image to the predicted class subfolder
        image_filename = f"incorrect_instance_{index}.png"
        image_path = os.path.join(predicted_class_folder, image_filename)

        # Assuming images[index] is a PyTorch tensor
        save_image(images[index], image_path)

# Run on holdout test data
def test(dataloader, model, loss_fn, model_weights, save_incorrect_images_local):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model = CNN().to(device)
    model.load_state_dict(model_weights)
    model.eval()
    test_loss, correct = 0, 0
    all_predictions, all_targets, all_images = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_predictions.append(pred)
            all_targets.append(y)
            all_images.append(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Log incorrect instances with images
    if save_incorrect_images_local:
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_images = torch.cat(all_images)

        save_incorrect_images(all_predictions, all_targets, all_images, class_labels_dict)

    # Calculate metrics
    num_classes = len(torch.unique(all_targets))
    metrics_df = calculate_metrics(all_predictions, all_targets, num_classes)

    # Log metrics to W&B
    wandb.log({"classification metrics": wandb.Table(dataframe=metrics_df)})

    # Calculate accuracy and loss
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>4f} \n")

    # Log accuracy and loss to W&B
    wandb.log({'test_acc': 100*correct, 'test_loss': test_loss})

    return None

test(test_dataloader, model, loss_fn, best_model_weights, save_incorrect_images_local)

# %%
# Finish wandb session
wandb.finish()
# %%
