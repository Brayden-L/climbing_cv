# %%
# Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ExponentialLR
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch_lr_finder import LRFinder
from torchvision.utils import save_image
import wandb
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from dotenv import dotenv_values
import os
from datetime import datetime
from tqdm.notebook import tqdm
from PIL import Image
import random
import math

##
# Seed freeze
seed=1220
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

##
# Run utility bools
project_name = "climb_classifier_rear_glory_topo_transferlearn2_cnn" # Name of project to send wandb logs to.
wandb_enabled = True # Whether to track this pytorch training with wandb.
wandb_grad_tracking = False # Whether to track gradients, can be computationally expensive, but good to track down exploding/vanishing gradients
save_best_model = True # Whether to save the best model locally.
save_incorrect_images_local = True # Whether to save incorrectly labelled test images locally, useful to manually look for trends in difficult to classify images
custom_verbosity = True # Whether to enable additional print messages that provide additional context to the training.

data_path = './climbing_classifier_data'

##
# wandb login
if wandb_enabled:
    wandb_enabled_str = 'online'
else:
    wandb_enabled_str = 'disabled'
    
os.environ["WANDB_API_KEY"] = dotenv_values(".env")['WANDB_API_KEY']
wandb.login()

##
# Default configruation or for single runs
default_config={'test_split': 0.1, # proportion of data is assigned to validation
            'val_split': 0.1, # proportion of data assigned to testing
            'pretrain_type': 'rt', # 'ffe (fixed feature extractor) or rt (retrain)
            'pretrain_model_select': 'vgg11', # any of available torchvision models
            'class_imbalance_split_handling': True, # Enables balanced splitting of train/val/test datasets
            'class_imbalance_sample_handling': True, # Enables a weighted random sampler when loading data into batches
            'class_imbalance_loss_handling': False, # Enables weighted application of losses based on class.
            'img_size': (224, 224), # WxH of image to transform to
            'complex_rand_image_transform_enabled': False, # Whether to apply more complex random image transformations such as rand_flips
            'batch_size': 32,
            'lr_basic_or_finder': 'finder', # 'basic' or 'finder'. Either use a user defined value or use a LRRT function to find the optimal value
            'basic_lr': 1e-7, # Only used if lr_basic_or_finder is basic
            'lr_schedule': 'none', # none, onecycle, exp_decay, cosan_wr
            'onecycle_factor': 10, # What proportion of starting value to rise to. float >1
            'exp_decay_factor': 1e-2, # What proportion of starting value to asymptote to float 0<x<1
            'cosan_wr_period': 10, # T0 for cosine w/ warm restart
            'cosan_wr_mult': 1, # T_mult for cosine w/ warm restart
            'weight_decay': 0, # Weight decay for optimizer
            'dropout': 0.5,
            'epochs': 30
        }

##
def main(input_config=default_config): # the sweep agent will override the default_config if you use it.
    # init wandb
    with(wandb.init(config=input_config, mode=wandb_enabled_str)):
    
        config=wandb.config
        # Store into vars for readability
        test_split = config['test_split']
        val_split = config['val_split']
        pretrain_type = config['pretrain_type']
        pretrain_model_select = config['pretrain_model_select']
        class_imbalance_split_handling = config['class_imbalance_split_handling']
        class_imbalance_sample_handling = config['class_imbalance_sample_handling']
        class_imbalance_loss_handling = config['class_imbalance_loss_handling'] 
        img_size = config['img_size']
        complex_rand_image_transform_enabled = config['complex_rand_image_transform_enabled']
        batch_size = config['batch_size']
        lr_basic_or_finder = config['lr_basic_or_finder']
        basic_lr = config['basic_lr']
        lr_schedule = config['lr_schedule']
        onecycle_factor = config['onecycle_factor']
        exp_decay_factor = config['exp_decay_factor']
        cosan_wr_period = config['cosan_wr_period']
        cosan_wr_mult = config['cosan_wr_mult']
        weight_decay = config['weight_decay']
        dropout = config['dropout']
        epochs = config['epochs']
        
        run_name = wandb.run.name

        ##
        # Training device selection
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        ##
        # Image transformation
        if complex_rand_image_transform_enabled: # more complex
            img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalization
            ])
        else: # Basic
            img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet normalization
            ])

        ##
        # Data loading
        full_dataset = datasets.ImageFolder(root=data_path, transform=img_transform)

        ##
        # Class labels, weights etc.
        
        # Basic class labels and info
        class_labels = full_dataset.classes
        num_classes = len(class_labels)
        class_labels_dict = {i: class_labels[i] for i in range(num_classes)}
        class_indices = {label: [] for label in class_labels}

        for idx, (_, label_idx) in enumerate(full_dataset.imgs):
            class_name = class_labels[label_idx]
            class_indices[class_name].append(idx)

        class_label_counts = {label: len(indices) for label, indices in class_indices.items()}

        if custom_verbosity:
            for label, count in class_label_counts.items():
                print(f"{label}: {count} samples\n")

        # Calculate class weights
        def calculate_class_weights(dataset):
            labels = np.array([label_idx for _, label_idx in dataset.imgs])
            class_weights = torch.Tensor(compute_class_weight('balanced', classes=np.unique(labels), y=labels)).to(device)
            return class_weights

        class_weights = calculate_class_weights(full_dataset)
        if custom_verbosity:
            print(f"Class Weights: {class_weights}")

        ##
        # Data Splitting into train, validation, and test sets
        if class_imbalance_split_handling:
            # Ensure that train, val and test all have balanced classes using sklearn train_test_split function. A bit bulky as a result.
            # Data splitting
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                full_dataset.imgs, full_dataset.targets, 
                test_size=val_split + test_split, 
                random_state=42, 
                stratify=full_dataset.targets
            )

            val_data, test_data, val_labels, test_labels = train_test_split(
                temp_data, temp_labels, 
                test_size=test_split / (val_split + test_split), 
                random_state=42, 
                stratify=temp_labels
            )

            # There is no generic empty dataset class so we make our own.
            class CustomImageDataset(Dataset):
                def __init__(self, data, labels, transform=None):
                    self.data = data
                    self.labels = labels
                    self.transform = transform

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    img_path, label = self.data[idx]
                    img = Image.open(img_path).convert("RGB")

                    if self.transform:
                        img = self.transform(img)

                    return img, label

            # Dataset setup
            train_dataset = CustomImageDataset(train_data, train_labels, transform=img_transform)
            val_dataset = CustomImageDataset(val_data, val_labels, transform=img_transform)
            test_dataset = CustomImageDataset(test_data, test_labels, transform=img_transform)
            
            # Dataloader setup
            if class_imbalance_sample_handling:
                # Create a WeightedRandomSampler
                sampler_labels = [label for _, label in train_data]
                sampler_weights = torch.tensor([class_weights[label] for label in sampler_labels], dtype=torch.float).to(device)
                sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(train_data), replacement=True) # this makes it more likely for an under-represented class to be sampled in a batch, the batch itself is not guaranteed to be balanced.
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            
        else:
            # If we want a basic split not accounting for class imbalance
            # Data splitting
            train_data, val_data, test_data = random_split(full_dataset, [1-(val_split+test_split), val_split, test_split])


            # Data loader setup
            if class_imbalance_sample_handling:
                # Create a WeightedRandomSampler
                sampler_labels = [label for _, label in train_data]
                sampler_weights = torch.tensor([class_weights[label] for label in sampler_labels], dtype=torch.float).to(device) # WeightedRandomSampler wants a weight tensor that is the length of the sampleset, with each entry being the samples class weight.
                sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(train_data), replacement=True) # this makes it more likely for an under-represented class to be sampled in a batch, the batch itself is not guaranteed to be balanced.
                train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            else:
                # Basic sampler
                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            # Assign the sampler to the dataloader
            val_dataloader = DataLoader(val_data, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # Data size confirmation
        for X, y in val_dataloader:
            num_img_channels = X.shape[1]
            final_img_height = X.shape[2]
            final_img_width = X.shape[3]
            if custom_verbosity:
                print(f"Shape of X [N, C, H, W]: {X.shape}")
                print(f"Shape of y: {y.shape} {y.dtype}")
            break

        ##
        # Model definition
        class PretrainModel(nn.Module):
            def __init__(self):
                super(PretrainModel, self).__init__()

                # Load pre-trained pretrained model
                if pretrain_model_select == 'alexnet':
                    self.pretrain_net = models.alexnet(pretrained=True)
                if pretrain_model_select == 'vgg11':
                    self.pretrain_net = models.alexnet(pretrained=True)
                if pretrain_model_select == 'vgg16':
                    self.pretrain_net = models.vgg16(pretrained=True)
                if pretrain_model_select == 'resnet18':
                    self.pretrain_net = models.resnet18(pretrained=True)
                if pretrain_model_select == 'resnet50':
                    self.pretrain_net = models.resnet50(pretrained=True)
                if pretrain_model_select == 'densenet121':
                    self.pretrain_net = models.densenet121(pretrained=True)
                if pretrain_model_select == 'squeezenet1_1':
                    self.pretrain_net = models.squeezenet1_1(pretrained=True)
                if pretrain_model_select == 'mobilenet_v3_large':
                    self.pretrain_net = models.mobilenet_v3_large(pretrained=True)
                else:
                    raise ValueError('pretrain_model_select var not valid')
                
                # Freeze the pretrained model parameters
                for param in self.pretrain_net.parameters():
                    if pretrain_type == 'ffe':
                        param.requires_grad = False
                    if pretrain_type == 'rt':
                        param.requires_grad = True
                    
                pretrain_final_out_channels = self.pretrain_net(torch.empty(X.shape)).size(-1) # calculates number of channels at end of pretrain model

                # Add custom fully connected layer on top
                self.classifier = nn.Sequential(
                    nn.Linear(in_features=pretrain_final_out_channels, out_features=600),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_features=600, out_features=120),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_features=120, out_features=num_classes) # Softmax is applied by the crossentropy loss function, so is not needed here.
                )

            def forward(self, x):
                features = self.pretrain_net(x) # Run it through the pretrain
                out = self.classifier(features) # Then the custom classifier
                return out

        model = PretrainModel().to(device)
        if custom_verbosity:
            print(model)

        ##
        # Loss function definition
        if class_imbalance_loss_handling:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights) # Loss scaled to class imbalance
        else:
            loss_fn = nn.CrossEntropyLoss()

        ##
        # optimizer definition and lr type

        # Define finder function
        # If no schedule, the LR that yields the most negative slope on the chart is to be used.
        # If yes to a schedule, the min_lr will be the lr with the most negative slope, and the max_lr will be 10x smaller than the lr at the minimum loss value.
        def lrfinder(model, weight_decay, loss_fn, device, train_dataloader):
            lower_lrf_bound = 1e-10
            upper_lrf_bound = 1
            lrf_numiter = 100

            optimizer = torch.optim.Adam(model.parameters(), lr=lower_lrf_bound, weight_decay=weight_decay) # Initialize basic optimizer
            
            lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
            lr_finder.range_test(train_dataloader, end_lr=upper_lrf_bound, num_iter=lrf_numiter, step_mode='exp')
            if wandb.run.sweep_id is None: # We typically only want to visibly check the plot as part of a singular run, not a sweep. This plot will also pause the sweep script, which we do not want.
                lr_finder.plot() # to inspect the loss-learning rate graph

            def suggested_lr(lr_finder, lower_lrf_bound):
                lrs = np.array(lr_finder.history["lr"])
                losses = np.array(lr_finder.history["loss"])

                max_lr_bound = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"])]
                max_lr_bound = 0.1*max_lr_bound # It is typical to take 10x less This is to effectively avoid the exploding loss portion of the plot.

                min_grad_idx = None
                try:
                    min_grad_idx = (np.gradient(np.array(losses))).argmin()
                    min_lr_bound = lrs[min_grad_idx]
                    print(f'optimized or min_lr: {min_lr_bound}')
                except ValueError:
                    print("Failed to compute the gradients, there might not be enough points. Output is lower_lrf_bound")
                    min_lr_bound = lower_lrf_bound
                print(f'max_lr: {max_lr_bound}')
                    
                return max_lr_bound, min_lr_bound
            
            max_scheduled_lr, min_scheduled_lr = suggested_lr(lr_finder, lower_lrf_bound) # optimized lr is either the single value past to a no-schedule setup, or the minimum value given to a schedule
            lr_finder.reset() # to reset the model and optimizer to their initial state
            
            return max_scheduled_lr, min_scheduled_lr

        # Basic assigned lr, no scheduler
        if lr_basic_or_finder == 'basic' and lr_schedule == 'none':
            scheduler_exists = False
            optimizer = torch.optim.Adam(model.parameters(), lr=basic_lr, weight_decay=weight_decay)

        # LR Finder, no scheduler
        if lr_basic_or_finder == 'finder' and lr_schedule == 'none':
            scheduler_exists = False
            
            _, min_scheduled_lr = lrfinder(model, weight_decay, loss_fn, device, train_dataloader)
            # Redefine optimizer with found LR
            optimizer = torch.optim.Adam(model.parameters(), lr=min_scheduled_lr, weight_decay=weight_decay)

        # Onecycle, using finder to determine the min_lr, or a defined value given by basic_lr. either way uses lrfinder to determine the max_lr.
        if lr_schedule == 'onecycle':
            scheduler_exists = True
            
            if lr_basic_or_finder == 'basic':
                onecycle_min_lr = basic_lr
                onecycle_max_lr = onecycle_factor*onecycle_min_lr
            if lr_basic_or_finder == 'finder':
                onecycle_max_lr, onecycle_min_lr = lrfinder(model, weight_decay, loss_fn, device, train_dataloader)
            
            # Define the number of iterations (epochs * len(train_dataloader))
            total_iterations = epochs * len(train_dataloader)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=onecycle_min_lr, weight_decay=weight_decay)
            scheduler = OneCycleLR(optimizer, max_lr=onecycle_max_lr, total_steps=total_iterations, anneal_strategy='cos', cycle_momentum=True)

        if lr_schedule == 'exp_decay':
            scheduler_exists = True
            
            gamma = (exp_decay_factor) ** (1/epochs)
            if lr_basic_or_finder == 'basic':
                exp_starting_lr = basic_lr
            if lr_basic_or_finder =='finder':
                _, exp_starting_lr = lrfinder(model, weight_decay, loss_fn, device, train_dataloader)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_starting_lr, weight_decay=weight_decay)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        if lr_schedule == 'cosan_wr':
            scheduler_exists = True
            
            if lr_basic_or_finder == 'basic':
                cosan_starting_lr = basic_lr
            if lr_basic_or_finder =='finder':
                _, cosan_starting_lr = lrfinder(model, weight_decay, loss_fn, device, train_dataloader)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=cosan_starting_lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cosan_wr_period, T_mult=cosan_wr_mult)

        ##
        # Early stopping logic
        
        # If val accuracy remains the same, then the model has likely diverged and is guessing a single class.
        # This utility function will end the run if it detects n epochs greater than the patience at the same val accuracy.
        loss_acc_es_patience = 5
        loss_acc_es_epochs_wo_imp = 0
        loss_acc_es_prev_val_acc = 0.0
        
        def val_acc_early_stop(val_accuracy, prev_val_accuracy, patience, epochs_wo_imp):
            decimal_places = 3
            rounded_val_acc = round(val_accuracy, decimal_places)
            rounded_prev_val_acc = round(prev_val_accuracy, decimal_places)
            if rounded_val_acc == rounded_prev_val_acc:
                epochs_wo_imp += 1
            else:
                epochs_wo_imp = 0
            loss_acc_es_bool = epochs_wo_imp == patience # This line looks a lil silly but sets the bool to true if the other to vars are equal.
            prev_val_accuracy = val_accuracy # Set current val_acc to previous for next check
            return loss_acc_es_bool, prev_val_accuracy, epochs_wo_imp


        ##
        # Training loop definition
        def train(dataloader, model, loss_fn, optimizer, epoch):
            model.train()
            
            size = len(dataloader.dataset)
            running_loss = 0.0
            
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                # Forward step
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                
                # Print some info every so often to provide some insight into how the training is going
                if batch % 5 == 0:
                    # Basic batch info
                    loss, current = loss.item(), (batch + 1) * len(X)
                    
                    # Class distribution for each batch, useful when troubleshooting class imbalance issues
                    batch_class_distribution = {label: 0 for label in class_labels}
                    for label in y.cpu().numpy():
                        class_name = class_labels[label]
                        batch_class_distribution[class_name] += 1
                    percentage_distribution_class_batch = {class_name: count / len(y) * 100 for class_name, count in batch_class_distribution.items()}
                    formatted_percentage_distribution_class_batch = {class_name: f"{percentage:.0f}" for class_name, percentage in percentage_distribution_class_batch.items()}


                    if custom_verbosity:
                        print(f"Sample Count: [{current:>5d}/{size:>5d}] | Loss: {loss:>4f} | Class Distro: {formatted_percentage_distribution_class_batch}")
                    else:
                        print(f"Sample Count: [{current:>5d}/{size:>5d}] | Loss: {loss:>4f}")
 

            
            # Calculate average loss for the epoch
            avg_loss = running_loss / len(dataloader)

            # Log average loss to wandb
            wandb.log({"Training Loss": avg_loss}, step=epoch)
                

        ##
        # Validation definition
        best_accuracy = 0.0
        best_model_weights = None
        best_epoch = None
        def val(dataloader, model, loss_fn, epoch, best_accuracy, best_model_weights):
            model.eval()
            
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            val_loss, correct = 0, 0
            
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= num_batches
            correct /= size
            val_acc = correct*100
            print(f"Val Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>4f} \n")
            
            wandb.log({"Val Loss": val_loss, "Val Accuracy": val_acc}, step=epoch)
                
            # Check if the current model is the best so far
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_weights = model.state_dict()
                best_epoch = epoch+1
                print(f'New Best Model: Epoch:{best_epoch} | Accuracy:{val_acc:>4f}\n')
            
            return best_accuracy, best_model_weights, val_acc

        ##
        # Run Train/Val
        if wandb_grad_tracking:
            wandb.watch(model, loss_fn, log='gradients', log_freq=25) # log_freq here is in terms of batches I'm pretty sure
            
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, epoch)
            best_accuracy, best_model_weights, val_acc = val(val_dataloader, model, loss_fn, epoch, best_accuracy, best_model_weights) # best accuracy must be fed back into the function
            if scheduler_exists:
                scheduler.step()
                
            # Early Stopping Logic
            loss_acc_es_bool, loss_acc_es_prev_val_acc, loss_acc_es_epochs_wo_imp = val_acc_early_stop(val_acc, loss_acc_es_prev_val_acc, loss_acc_es_patience, loss_acc_es_epochs_wo_imp)
            if loss_acc_es_bool:
                wandb.log({'loss_acc_es_bool': loss_acc_es_bool})
                print('Validation Accurracy Stalled, Early Stopping Triggered')
                break
        print("Training Complete")
        
        ##
        # Save the best model locally
        if save_best_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saved_models/{run_name}_{timestamp}.pth"
            torch.save(best_model_weights, filename)
            print(f'Epoch {best_epoch} saved as {filename}')
            
        ##
        # Test Data
        # Utility funcs for test data output
        def calculate_metrics(predictions, targets, num_classes, class_labels):
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
            inc_img_dir = f"{run_name}_{timestamp}"
            
            # Create the timestamped directory
            save_dir = os.path.join(base_dir, inc_img_dir)
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
        if best_model_weights is not None: # The test will error if we were unable to get any model due to upstream error
            def test(dataloader, model, loss_fn, model_weights, save_incorrect_images_local):
                size = len(dataloader.dataset)
                num_batches = len(dataloader)
                
                model = PretrainModel().to(device)
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

                all_predictions = torch.cat(all_predictions)
                all_targets = torch.cat(all_targets)
                all_images = torch.cat(all_images)

                # Log incorrect instances with images
                if save_incorrect_images_local:
                    save_incorrect_images(all_predictions, all_targets, all_images, class_labels_dict)

                # Calculate metrics
                metrics_df = calculate_metrics(all_predictions, all_targets, num_classes, class_labels)

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

        # Empty cache after run has completed so memory does not run out run-to-run
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    main()
# %%