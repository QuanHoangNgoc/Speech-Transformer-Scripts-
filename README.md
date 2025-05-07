# Speech2Text Project

## Overview
The Speech2Text project is designed to train a speech recognition model using a transformer architecture. This project leverages PyTorch for deep learning and includes functionalities for data preprocessing, model training, validation, and evaluation.

## Features
- Transformer-based speech recognition model.
- Data preprocessing utilities for loading and preparing datasets.
- Training and validation routines with logging.
- Learning rate scheduling.
- Reproducibility through random seed setting.

## Project Structure
The project is organized as follows:
- **`train_model.py`**: Contains the main logic for training the speech recognition model, including data loading, model initialization, and training loop.
- **`preprocessor.py`**: Includes functions for loading and preprocessing the dataset.
- **`score.py`**: Implements functions for calculating evaluation metrics such as loss and Word Error Rate (WER).
- **`speech_transformer.py`**: Defines the architecture of the speech transformer model.
- **`requirements.txt`**: Lists the required Python packages for the project.
- **`points/`**: Directory where model checkpoints are saved during training.
- **`NP/`**: contains all feature-npy files.
- **`metadata/`**: train.json, dev.json, test.json 

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Speech2Text
pip install -r requirements.txt
```

Could you ensure you have installed Python 3.6 or higher and the necessary libraries?

## Usage
- To run the training process, execute the following command:
```bash
python train_model.py
```

To build feature-npy files, check the code at:
```bash
python convert_to_np.py
```

## Setting 
You can modify the training parameters directly in the `train_model.py` file, such as:
- Number of epochs
- Batch size
- Learning rate
- Warmup steps

### Training
The training process involves the following steps:
1. **Data Loading**: The data is loaded using the `preprocessor` module.
2. **Model Initialization**: The model is initialized with specified parameters.
3. **Training Loop**: The model is trained over a specified number of epochs, with logging of loss and learning rate.

### Evaluation
After training, the model can be evaluated using the validation dataset. The evaluation metrics include:
- Average loss
- Word Error Rate (WER)


