# Cascade Learning in Multi-task Encoder-Decoder Networks

This repository shares the code of two novel convolutional neural networks (CNNs) developed for the article **"Cascade learning in multi-task encoder-decoder networks for concurrent bone segmentation and glenohumeral joint clinical assessment in shoulder CT scans."**

The repository is organized into two main folders:

- **AI-Shoulder/**: Contains the neural network architecture and utility functions.
  - `__init__.py`: Initialization script for package setup.
  - `cnn.py`: Code for the CNN architectures, including both segmentation and classification networks.
  - `utils.py`: Contains helper functions used throughout the codebase (e.g., data preprocessing, metrics computation).
  
- **images/**: Contains images of the two networks used in the article for illustration purposes.

# CEL-UNet

The CEL-UNet model was designed and developed to boost the segmentation quality on bone CT scans and enable preoperative planning in PSI-based interventions. The identification of severe pathological anatomy deformations and narrow joint space was enhanced by combining traditional semantic mask segmentation with boundary identification. In detail, the decoder was split into two parallel branches, one devoted to region segmentation (RA, region-aware branch) and the other addressing edge detection (CA, contour-aware branch). Vertical unidirectional skip connections were added between each CA branch processing block and its corresponding one in the RA branch, to integrate region and boundary-related features at increasing resolution. A specialized processing module was also implemented after each CA branch layer, using the pyramidal edge extraction (PEE).

![CEL-UNet architecture](images/CEL-UNet.png)

# Arthro-Net

The Arthro-Net is a novel CNN architecture designed for multi-task, multi-class staging of GH osteoarthritic-related conditions. The network processes GH-centered patches and classifies the severity of each pathology according to predefined categories: three classes for osteophyte size (OS), three classes for joint space narrowing (JS), and two classes for the humeroscapular alignment (HSA). The Arthro-Net encoder is a sequence of convolutional/downsampling blocks to extract features at decreasing resolution up to the bottleneck. Each processing block embeds two sequences of convolutional with linear activation, batch normalization, and ReLU activation layers. The filter size and stride length are 3×3×3 and 1×1×1, respectively. This configuration is consistent with the CEL-UNet feature extractor, which captures relevant osseous characteristics from CT images. However, the number of feature maps in the Arthro-Net doubles every two processing blocks, to avoid unnecessary over-parametrization. Downsampling is performed through max-pooling layers with filter size and stride length of 2×2×2. According to state-of-the-art classification architecture, a flatten layer was placed after the bottleneck to reshape the data into a one-dimensional tensor while preserving the total number of elements. Three separate fully connected branches, characterized by two consecutive dense layers with input sizes of 256 and 32, and a ReLU activation function, were added to specialize the network for each classification task.

![Arthro-Net architecture](images/Arthro-Net.png)

## Repository Structure

```bash
.
├── AI-Shoulder
│   ├── __init__.py
│   ├── cnn.py
│   ├── utils.py
│
├── images
│   ├── CEL-UNet.png
│   ├── Arthro-Net.png
│
├── .gitignore
├── README.md
├── setup.py
    

