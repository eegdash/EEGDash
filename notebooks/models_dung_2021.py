import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch

def create_model_vgg16(input_shape=(1, 1, 24, 24)):
    subsample = 4
    tmp = models.vgg16()
    tmp.features = tmp.features[0:17]  # truncate features

    # Build modified features block
    feature_modules = []
    for layer in tmp.features.children():
        if isinstance(layer, nn.Conv2d):
            # Change channel count
            if layer.in_channels == 3:
                in_channels = 1
            else:
                in_channels = int(layer.in_channels / subsample)
            out_channels = int(layer.out_channels / subsample)
            feature_modules.append(
                nn.Conv2d(in_channels, out_channels, layer.kernel_size, layer.stride, layer.padding)
            )
        else:
            feature_modules.append(layer)
    features = nn.Sequential(*feature_modules)
    
    # Create a sequential container for the entire model
    model_seq = nn.Sequential()
    model_seq.add_module('features', features)
    model_seq.add_module('flatten', nn.Flatten())
    
    # Determine the flattened feature size using a dummy input
    with torch.no_grad():
        dummy_input = torch.zeros(input_shape)
        flatten_size = model_seq.features(dummy_input).view(dummy_input.size(0), -1).shape[1]
    print("Computed flattened feature size:", flatten_size)
    
    # Build modified classifier block
    classifier_modules = []
    for layer in tmp.classifier.children():
        if isinstance(layer, nn.Linear):
            # For the first Linear layer, use the computed flattened feature size
            if layer.in_features == 25088:
                in_features = flatten_size
            else:
                in_features = int(layer.in_features / subsample)
            # For the last Linear layer, change the output to 2 classes
            if layer.out_features == 1000:
                out_features = 2
            else:
                out_features = int(layer.out_features / subsample)
            classifier_modules.append(nn.Linear(in_features, out_features))
        else:
            classifier_modules.append(layer)
    classifier = nn.Sequential(*classifier_modules)
    model_seq.add_module('classifier', classifier)
    
    return model_seq

def create_model_original_129_614():
    '''
    Create the CNN following configuration in van Putten et al. (2018)
    '''
    model = nn.Sequential(
            nn.Conv2d(1,100,3),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(4,2)),
            nn.Dropout(0.25),
            nn.Conv2d(100,100,3),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), stride=(4,2)),
            nn.Dropout(0.25),
            nn.Conv2d(100,300,(2,3)),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), stride=4),
            nn.Dropout(0.25),
            nn.Conv2d(300,300,(1,7)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(300,100,(1,3)),
            nn.ReLU(),
            nn.Conv2d(100,100,(1,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2600,6144),
            nn.ReLU(),
            nn.Linear(6144,2),
        )
    return model


def create_model_original_24_256():
    '''
    Create the CNN following configuration in van Putten et al. (2018)
    '''
    model = nn.Sequential(
            nn.Conv2d(1,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,300,(2,3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(300,300,(1,7)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(300,100,(1,3)),
            nn.ReLU(),
            nn.Conv2d(100,100,(1,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1900,6144),
            nn.ReLU(),
            nn.Linear(6144,2),
        )
    return model