# Multi-Character OCR Project

This repository contains the code and pretrained models developed for a course project on Optical Character Recognition (OCR). The focus of this project was on multi-character OCR using convolutional neural networks (CNN) and bidirectional Long Short-Term Memory (BiLSTM) layers for sequence modeling. The repository includes the implementation of the model, pretrained weights, and additional resources for further exploration.

## Repository Structure

```
|-- MultiOCR_Project.ipynb       # Main notebook containing the OCR model implementation and training code
|-- multi_ocr_model.pth     # Pretrained weights for the multi-character OCR model
|-- README.md               # Documentation for the repository
```

## Contents

### 1. `MultiOCR_Project.ipynb`
This Jupyter notebook includes:
- The implementation of the multi-character OCR model.
- Code for training and evaluating the model.
- Visualization of training progress, including accuracy and loss graphs.

### 2. `multi_ocr_model.pth`
This file contains the pretrained weights for the multi-character OCR model. The model was trained using the following configuration:
- Batch Size: 64
- Number of Epochs: 15
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Connectionist Temporal Classification (CTC) Loss

The model achieves:
- **Character Accuracy:** 87.76% on the test set
- **Sequence Accuracy:** 52.58% on the test set
- **Test Loss:** 0.3549

## How to Use

### Requirements
The project requires the following libraries:
- Python 3.8+
- PyTorch 1.12+
- torchvision
- matplotlib
- tqdm
- Jupyter Notebook

### Steps to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook OCR_Project.ipynb
   ```

4. Load the pretrained model by updating the checkpoint path in the notebook:
   ```python
   model.load_state_dict(torch.load('multi_ocr_model.pth'))
   ```

5. Run the cells to test the model or retrain it.


6Evaluate the model on your dataset and observe the results.

## Results and Insights
- The **multi-character OCR model** demonstrates strong character-level accuracy, achieving 87.76% on the test set.
- Sequence accuracy is relatively lower (52.58%), suggesting room for improvement in capturing dependencies across longer sequences.
- Further exploration into attention mechanisms or transformer-based architectures is recommended to enhance sequence modeling.

## Challenges Encountered
- Data preparation for multi-character sequences required significant preprocessing.
- Training for high accuracy demanded extended computational time.

## Future Work
- Integrating more robust sequence modeling techniques, such as transformers or hybrid models.
- Experimenting with noisy datasets to enhance robustness.
- Comparing performance against simpler baseline models to evaluate improvements.
