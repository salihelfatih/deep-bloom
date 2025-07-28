# 🌸 Deep Bloom

**Deep Bloom** is a deep learning project that classifies flowers from images using a ResNet-based neural network and transfer learning. Built with **PyTorch** and deployed with **Gradio**, this app blends the elegance of nature with the power of machine learning.

> Upload a flower picture. Watch the model guess. Let it bloom!

## 🔍 Overview

- 🌼 102 Flower Classes  
- 🧠 Transfer Learning with ResNet  
- 🧪 Accuracy + Confusion Matrix Evaluation  
- 🖼️ Live Prediction via Gradio Interface  
- 🚀 Deployed on Hugging Face Spaces

## 🧠 Model Architecture

- Pretrained **ResNet** (ImageNet)
- Frozen convolutional base
- Custom classifier head (102 outputs)
- Trained with **CrossEntropyLoss** + **Adam optimizer**

## 📊 Dataset

- **Name**: Flowers102  
- **Source**: [Torchvision Datasets](https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html)  
- **Size**: 8,189 images across 102 flower species
- **Use Case**: Fine-grained image classification

## 🧠 Model Details

- **Backbone**: Pretrained ResNet (ImageNet)
- **Strategy**:  
  - Freeze feature extractor  
  - Replace final layer → 102 classes  
  - Train classifier head  
- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam  
- **Framework**: PyTorch

## 🎯 Evaluation Metrics

- Top-1 Accuracy
- Confusion Matrix (per class)
- Weighted F1 Score
- Visual Debugging (optional Grad-CAM)

## 🌐 Web Application

Gradio makes the model accessible in a few clicks.

**App Features:**
- Upload an image
- Get predicted flower name
- Clean, intuitive UI with floral-themed styling

## 🖼️ Live Demo

> [👉 Try it on Hugging Face Spaces](https://huggingface.co/spaces/salihelfatih/deep-bloom)

## 🎨 Creative Vision

*Deep Bloom* is more than a technical project. It's an exploration of how neural networks can learn to see beauty.  
It combines machine learning with artistic curiosity, delivering a tool that's practical, educational, and visually delightful.

## ✅ Roadmap

- [x] Load and preprocess Flowers102
- [x] Train classifier using transfer learning
- [x] Evaluate with visual and metric-based tools
- [x] Build interactive Gradio app
- [x] Deploy on Hugging Face Spaces
- [ ] Add Grad-CAM for interpretability
- [ ] Improve UI with animations or filters

## 📁 Project Structure

```plaintext
deep_bloom/
├── deep_bloom_core/
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── app/
│   └── interface.py
│
├── model.pth
├── requirements.txt
└── README.md
```

## 🧾 License

MIT License — free to use, remix, and bloom!

## 🙌 Credits

Developed by Salih Elfatih as a capstone project on deep learning and computer vision.
Flowers bloom. So should code!
