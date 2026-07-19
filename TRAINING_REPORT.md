# Model Training & Evaluation Report

## 1. Architecture & Approach
To achieve highly accurate instance segmentation, this project utilizes a **Mask R-CNN** architecture with a ResNet-50-FPN backbone (implemented via PyTorch/Torchvision). This avoids off-the-shelf wrappers like YOLO, allowing fine-grained control over the training loop, loss metrics, and inference pipeline.

## 2. Dataset Preparation
*   **Format:** COCO JSON format.
*   **Handling:** A custom PyTorch `Dataset` class was engineered to parse the COCO annotations, extract bounding boxes, and dynamically generate bitmasks.
*   **Data Integrity:** The data loader includes a filtering mechanism to automatically drop unannotated images, ensuring clean gradients during backpropagation.

## 3. Training Dynamics
*   **Optimizer:** Stochastic Gradient Descent (SGD)
*   **Learning Rate:** 0.005 (with momentum = 0.9, weight decay = 0.0005)
*   **Epochs:** 10
*   A custom training loop was utilized to log the loss dictionary (classifier loss, box regression loss, mask loss, and objectness loss). 

## 4. Quantitative Evaluation (Validation Set)
Standard COCO metrics were computed using the `torchmetrics` library on a held-out test set. 

| Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP @ 0.5:0.95** | `[Insert Score]` | Mean Average Precision across strict IoU thresholds. |
| **mAP @ 0.5** | `[Insert Score]` | Standard MAP at 50% bounding box overlap. |
| **Recall (mAR)** | `[Insert Score]` | Model's ability to detect all target instances. |

## 5. Visual Output
During inference, the model operates in `model.eval()` mode (without gradients). The raw predictions are parsed, filtered by a >50% confidence threshold, and visualized using OpenCV (Green bounding boxes and Red masks). 
*Demo images are available in the `inference_output` directory.*