import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define Generator
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

# Load model
device = 'cpu'
G = Generator()
G.load_state_dict(torch.load('generator.pth', map_location=device))
G.eval()

st.title("🧠 MNIST Digit Generator")

# Seed slider for controlled randomness
seed = st.slider("Choose a random seed", min_value=0, max_value=9999, value=42, step=1)

# Number of images
num_images = st.slider("How many images to generate?", min_value=1, max_value=10, value=5)

# Optional: single-digit or random digits
digit_mode = st.radio("Digit mode", ("Single Digit", "Random Digits"))

if digit_mode == "Single Digit":
    digit = st.selectbox("Select a digit (0–9)", list(range(10)))
    labels = torch.tensor([digit] * num_images)
else:
    labels = torch.randint(0, 10, (num_images,))

if st.button("Generate Images"):
    torch.manual_seed(seed)
    z_dim = 100
    z = torch.randn(num_images, z_dim) * 1.5
    label_noise = torch.randn_like(G.label_emb(labels)) * 0.2
    emb = G.label_emb(labels) + label_noise
    x = torch.cat([z, emb], dim=1)
    with torch.no_grad():
        generated_imgs = G.model(x).view(-1, 1, 28, 28).squeeze().numpy()

    fig, axs = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
    if num_images == 1:
        axs = [axs]  # Ensure iterable

    for i in range(num_images):
        axs[i].imshow(generated_imgs[i], cmap='gray')
        axs[i].set_title(f"Digit: {labels[i].item()}")
        axs[i].axis('off')

    st.pyplot(fig)
