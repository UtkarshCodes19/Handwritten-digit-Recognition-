# Handwritten Digit Recognizer using CNN

A beginner-friendly computer vision project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) and the MNIST dataset.

## Objective
Build an image-classification model that receives a handwritten digit image and predicts which digit it represents.

## ECE Connection
This project can be viewed as an intelligent embedded-vision pipeline:

**Image Sensor/Camera → Digital Image → Preprocessing → CNN → Embedded Processor/FPGA → Display/Control**

It connects digital image processing, embedded systems and machine learning. The software version in this repository focuses on the CNN algorithm; deployment to an FPGA, SoC or edge-AI device is a possible hardware extension.

## Technologies
- Python
- TensorFlow/Keras
- CNN
- MNIST
- NumPy
- Streamlit
- Matplotlib

## Model Architecture
1. Input: 28×28 grayscale image
2. Convolution layer
3. Max-pooling
4. Convolution layer
5. Max-pooling
6. Flatten
7. Dense layer
8. 10-class softmax output

## How to Run

### 1. Create a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the CNN
```bash
python train.py
```
This downloads MNIST through Keras, trains the model, evaluates it and creates:
- `digit_model.keras`
- `training_accuracy.png`
- `sample_predictions.png`

### 4. Run the web application
```bash
streamlit run app.py
```

## Expected Result
The model should normally achieve high test accuracy after a few epochs. Exact accuracy depends on the TensorFlow version, hardware and training configuration.

## Project Workflow
**Dataset → Preprocessing → CNN Training → Evaluation → Image Upload → Prediction**

## Future Improvements
- Add a drawing canvas instead of image upload.
- Deploy the model on an edge device.
- Quantize the CNN for low-power hardware.
- Implement inference on an FPGA or embedded AI platform.
- Add a camera-based real-time recognition mode.

## Disclaimer
This is an educational internship project. The model is intended for demonstration and learning.
