import numpy as np

import torch
import torch.nn as nn
from torchvision import models

class HyperMMPretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 5

        self.pretrained_model = models.vgg11(pretrained=True)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        self.pretrained_model.eval()

        self.encoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1)
        )
        self.decoder = nn.ConvTranspose2d(512, 512, kernel_size=1)
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))

        self.total_params = self.encoder[-1].weight.nelement() + self.encoder[-1].bias.nelement()
        self.embedding = nn.Embedding(20, self.emb_dim)
        
        self.hypernet = nn.Sequential(
            nn.Linear(self.emb_dim, 20), 
            nn.ReLU(), 
            nn.Linear(20, self.total_params)
        )

    def sample_weights(self, c):
        weights = self.hypernet(self.embedding(c))
        weights = weights.reshape([-1])

        next_idx = 0   
        layer = self.encoder[-1]
        if isinstance(layer, nn.Conv2d):
            # Get the current and next layer's neuron counts for the splice
            cur_idx = next_idx
            weight_size = np.prod(layer.weight.shape)
            bias_size = np.prod(layer.bias.shape)
            next_idx += weight_size + bias_size

            weight_splice = weights[cur_idx : cur_idx + weight_size].reshape(layer.weight.shape)
            bias_splice = weights[cur_idx + weight_size : next_idx].reshape(layer.bias.shape)

            # Copy over the generated weights into the parameters of the dynamics network
            # Note that this delete is important to properly establish the computation graph link
            del layer.weight
            layer.weight = weight_splice
            del layer.bias
            layer.bias = bias_splice


    def forward(self, x, c):
        self.sample_weights(c)
        x = torch.squeeze(x, dim=0) 
        
        input_features = self.pretrained_model.features(x)
        pooled_features = torch.max(input_features, 0, keepdim=True)[0]

        features = self.encoder(pooled_features)
        output_features = self.decoder(features)
        output_classif = self.classifier(torch.flatten(features, 1))

        return features, pooled_features, output_features, output_classif


class HyperMMNet(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

        self.phi = feature_extractor
        for param in self.phi.parameters():
            param.requires_grad = False

        self.rho = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )  

    def forward(self, X):
        feats = []
        for x in X:
            img = torch.squeeze(x[0], dim=0) 
            c = x[1]
            features,_,_,_ = self.phi(img, c)
            feats.append(torch.flatten(features, 1))

        latent = torch.stack(tuple(feats), 0)

        sum_latent = latent.mean(dim=0)
        output = self.rho(sum_latent)
        return output
    
"""
class HyperMMNet(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.phi = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256 * 6 * 6),
            #nn.ReLU(inplace=True),
            #nn.Linear(256 * 6 * 6, 256 * 6 * 6),
            nn.ReLU(inplace=True)
        )

        self.rho = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )  

    def forward(self, X):
        feats = []
        for x in X:
            img = torch.squeeze(x[0], dim=0) 
            c = x[1]
            features,_,_ = self.feature_extractor(img, c)
            feats.append(torch.flatten(features, 1))

        feats = torch.cat(tuple(feats), 0)

        latent = self.phi(feats)
        sum_latent = latent.mean(dim=0)
        output = self.rho(sum_latent)
        return output
"""