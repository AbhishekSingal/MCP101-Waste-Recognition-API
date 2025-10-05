# ♻️ Waste Image Recognition API

> **Waste Image Recognition** is a Python-based API that detects and classifies **wet vs dry waste** from images using computer vision and deep learning.  
> Developed as part of a team project for a **Sem1 MCL101 course**, this module handles the AI part of the waste segregation machine.

---

## 🖼️ Overview

This project focuses on the **image recognition component** of a waste segregation system.  
The API takes an input image from a camera and predicts whether the waste is **wet** or **dry**, providing real-time feedback for automated sorting systems.  

> ⚠️ **Note:** The `model_weights` file (~890 MB) is **not included** in this repository due to size. Please obtain it separately or contact the author.

---

## 🚀 Features

- 📸 **Real-Time Image Processing**  
  Captures images from a connected camera and pre-processes them for inference.

- 🤖 **Deep Learning-Based Classification**  
  Uses a pre-trained convolutional neural network (CNN) to classify waste images.

- 🧰 **Modular Python API**  
  Python scripts handle loading the model, predicting labels, and integrating with the hardware interface.

- 🔄 **Integration-Ready**  
  Can be plugged into robotic or mechanical systems for automatic sorting.

---

## ⚙️ Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** OpenCV, TensorFlow / PyTorch (depending on model implementation)
- **Used Wireless Video Streaming**
- **Paradigm:** Modular and object-oriented code

---
