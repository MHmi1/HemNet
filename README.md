# HemNet 🧹🎶📷  
**An End-2-End Diffusion & Autoencoder-Based Denoising Framework for Speech and Images**  

## 🌟 Overview  
HemNet is an open-source project designed to **denoise speech and images** using advanced **diffusion models and autoencoders**. Leveraging the power of deep learning, HemNet aims to enhance signal clarity by removing unwanted noise while preserving essential details.  

## 🚀 Features  
- **Diffusion-Based Denoising** – Progressive noise removal using diffusion models.  
- **Autoencoder-Based Restoration** – Efficient encoding-decoding for clean signal reconstruction.  
- **Customizable & Extendable** – Easy to modify for different datasets and noise types. 

## 🔬 Applications  
- **Generate syntactic noisy data** – You can easily select noise type and its amount to add to your dataset 
- **Speech Enhancement** – Noise removal from audio recordings, podcasts, and calls.  
- **Image Restoration** – Denoising photos, medical images, and low-light visuals.  
- **Preprocessing for AI Models** – Improve input data quality for ML pipelines.  


## 🛠 Installation  

git clone https://github.com/yourusername/HemNet.git  
cd HemNet  
pip install -r requirements.txt

## 📌 Usage  
### **Speech Denoising**  
from hemnet.speech import SpeechDenoiser  
denoiser = SpeechDenoiser(model="autoencoder")  
clean_audio = denoiser.denoise("noisy_audio.wav")  


