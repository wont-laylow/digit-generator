import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Define Generator model
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_size=28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_emb(labels)], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load the pre-trained generator model
device = 'cpu'
G = Generator()

if os.path.exists('generator.pth'):
    G.load_state_dict(torch.load('generator.pth', map_location=device))
    G.eval()
else:
    st.error("Model file 'generator.pth' not found. Please place it in the same directory as this script.")
    st.stop()

# Streamlit UI
st.title("MNIST Digit Generator")

digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate 5 Images"):
    z_dim = 100
    z = torch.randn(5, z_dim) + 0.05 * torch.randn(5, z_dim)  # Add jitter
    labels = torch.tensor([digit] * 5)

    # Add small noise to label embedding
    emb = G.label_emb(labels) + 0.05 * torch.randn_like(G.label_emb(labels))
    x = torch.cat([z, emb], dim=1)

    with torch.no_grad():
        generated_imgs = G.model(x).view(-1, 1, 28, 28).squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(generated_imgs[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
