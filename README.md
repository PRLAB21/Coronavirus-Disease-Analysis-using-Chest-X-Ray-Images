# Coronavirus Disease Analysis using Chest X-Ray Images and a Novel Deep Convolutional Neural Network

The recent emergence of a highly infectious and contagious respiratory viral disease known as COVID-19 has vastly impacted human lives and greatly burdened the health care system. Therefore, it is indispensable to develop a fast and accurate diagnostic system for timely identification of COVID-19 infected patients and to control its spread. This work proposes a new X-ray based COVID-19 classification framework consisting of (i) end-to-end classification module and (ii) deep feature-space based Machine Learning classification module. In this regard, two new custom CNN architectures, namely COVID-RENet-1 and COVID-RENet-2 are developed for COVID-19 specific pneumonia analysis by systematically employing Region and Edge base operations along with convolution operations. The synergistic use of Region and Edge based operations explores the region homogeneity, textural variations and region boundary; it thus helps in capturing the pneumonia specific pattern. In the first module, the proposed COVID-RENets are used for end-to-end classification. In the second module, the discrimination power is enhanced by jointly providing the deep feature hierarchies of the COVID-RENet-1 and COVID-RENet-2 to SVM for classification. The discrimination capacity of the proposed classification framework is assessed by comparing it against the standard state-of-the-art CNNs using radiologist’s authenticated chest X-ray dataset. The proposed classification framework shows good generalization (accuracy: 98.53%, F-score: 0.98, MCC: 0.97) with considerable high sensitivity (0.99) and precision (0.98). The exemplary performance of the classification framework suggests its potential use in other X-ray imagery based infectious disease analysis.

In this repository, we provide the MATLAB GUI and Testing Code for the Coronavirus Disease Analysis using Chest X-ray Images for the research community to use our research work.

## Overview of the workflow for the proposed COVID-19 Classification Framework

In this work, a new classification framework is developed based on deep learning and classical ML techniques for automatic discrimination of COVID-19 infected patients from healthy individuals based on chest X-ray images. The proposed classification framework is constituted of two modules: (i) Proposed COVID-RENet based end-to-end Classification, and (ii) Deep Concatenated Feature-space based ML classification. In the experimental setup, initially, training samples were augmented to improve the generalization. These augmented samples were used to train the two proposed modules. Fig. 1. (A) shows the modules of the proposed COVID-19 classification framework, whereas (B) gives the detailed overview of the workflow.

![Overview of the Workflow](./repo-images/workflow.png "Overview of the Workflow")

## Models Architectures

### Architectural details of the proposed COVID-RENet-1

![Architectural details of the proposed COVID-RENet-1](./repo-images/architecture-covid-renet-1.png "Architectural details of the proposed COVID-RENet-1")

### Architectural details of the proposed COVID-RENet-2

![Architectural details of the proposed COVID-RENet-2](./repo-images/architecture-covid-renet-2.png "Architectural details of the proposed COVID-RENet-2")

### Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification module

![Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification module](./repo-images/architecture-covid-19-classification-module.png "Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification module")

**Trained Model is available at [COVID-RENet-1](https://drive.google.com/file/d/1IY8Di0Jqlmb7pjw6OmKdmc2QasnLQ3sA/view?usp=sharing) and [COVID-RENet-2](https://drive.google.com/file/d/1ctjUFQLtNgMcKbQCYdPaPEsWXiBqhujM/view?usp=sharing) links.**

## Dataset

We built a new dataset consisting of X-ray images of COVID-19 pneumonia and healthy individuals in this work. X-ray images were collected from Open Source GitHub repository and Kaggle repository called “pneumonia”

**Dataset will be available on request, contact at <hengrshkhan822@gmail.com>**

### Dataset Samples

Panel (A) and (B) show COVID-19 infected and healthy images, respectively.

![Dataset Samples](./repo-images/dataset.png "Dataset Samples")

## Training plot of the proposed COVID-RENet-1 and COVID-RENet-2

|                                                                                                       |                                                                                                       |
| :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| ![Training Plot COVID-RENet-1](./repo-images/training-plot-RENet-1.png "Training Plot COVID-RENet-1") | ![Training Plot COVID-RENet-2](./repo-images/training-plot-RENet-2.png "Training Plot COVID-RENet-2") |

## Results

Performance comparison of the proposed COVID-RENet-1, COVID-RENet-2 and DCFS-MLC with standard existing CNNs.

![Performance-Comparison-01](./repo-images/performance-comparison-01.png "Performance-Comparison-01")

Detection and misclassification rate analysis of the proposed COVID-RENet-1, COVID-RENet-2, DCFS-MLC and ResNet.

![Performance-Comparison-02](./repo-images/performance-comparison-02.png "Performance-Comparison-02")

Performance metrics for the state-of-the-art CNN models that are trained from scratch and TL-based fine-tuned pre-trained on the augmented dataset.

![Results-01](./repo-images/results-01.png "Results-01")

Performance metrics for the deep feature extraction from custom layers of state-of-the-art training from scratch and TL-based fine-tuned pre-trained CNN on the augmented dataset.

![Results-02](./repo-images/results-02.png "Results-02")

COVID-19 (panel a & b) and Healthy (panel c & d) images, which are misclassified.

![output-01](./repo-images/output.png "output-01")

## PCA Visualization

Feature visualization of the proposed COVID-RENet-1, COVID-RENet-2, DCFS-MLC and the best performing standard existing CNN (ResNet) on test dataset.

|                                                                               |                                                                               |
| :---------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
|        ![PCA-DCFS-MLC](./repo-images/PCA-DCFS-MLC.png "PCA-DCFS-MLC")         | ![PCA-COVID-RENet-1](./repo-images/PCA-COVID-RENet-1.png "PCA-COVID-RENet-1") |
| ![PCA-COVID-RENet-2](./repo-images/PCA-COVID-RENet-2.png "PCA-COVID-RENet-2") |           ![PCA-ResNet](./repo-images/PCA-ResNet.png "PCA-ResNet")            |

## Heatmaps

Panel (a) shows the original chest X-ray image. Panel (b) shows the radiologist defined COVID-19 infected regions highlighted by yellow circle or black arrow. The resulted heat map of the proposed COVID-RENet-2 and COVID-RENet-1 is shown in panels (c & d), respectively. Panel (e) shows the heat map of the standard existing ResNet model.

![Heatmap](./repo-images/heatmap.png "Heatmap")

## ROC Curves

ROC curve for the proposed approach (DCFS-MLCS), proposed models (COVID-RENet-1 and COVID-RENet-2), and standard existing CNN models. The values in square bracket show a standard error at the 95% confidence interval.

|                                                                                                                |                                                                                                                         |
| :------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
|       ![ROC-trained-from-scratch](./repo-images/ROC-trained-from-scratch.png "ROC-trained-from-scratch")       |                 ![ROC-existing-CNN-SVM](./repo-images/ROC-existing-CNN-SVM.png "ROC-existing-CNN-SVM")                  |
| ![ROC-TL-based-fine-tuned-CNNs](./repo-images/ROC-TL-based-fine-tuned-CNNs.png "ROC-TL-based-fine-tuned-CNNs") | ![ROC-TL-based-fine-tuned-CNN-SVM](./repo-images/ROC-TL-based-fine-tuned-CNN-SVM.png "ROC-TL-based-fine-tuned-CNN-SVM") |

## Requirements

1. Matlab 2019b.
2. Deep Learning library.
3. NVIDIA GeForce GTX Titan X Computer.

## Setup

1. Clone this repo.

```git bash
git clone https://github.com/PRLAB21/Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images.git
```

2. Download model and place it in following structure.

```text
Coronavirus-Disease-Analysis-using-Chest-X-Ray-Images
|__ models
   |__ net_RENet_VGG_Modified1.mat
   |__ net_RENet_VGG_Modified2.mat
```

3. Testing images are downloaded along with this repo and are present inside "test-dataset" directory.

4. Run testing code using below mentioned methods.

## Inference Code

1. Open MATLAB.
2. Change MATLAB Working Directory to this repository's folder from top panel.
3. Now add each folder to MATLAB path from Current Folder panel by right clicking on each folder and selecting Add to Path > Selected Folder and Subfolders.
4. Run any of the two test model using following files.

-   **test_code_RENet_VGG_Modifier1.m**: Use this file for testing the model "net_RENet_VGG_Modified1".

-   **test_code_RENet_VGG_Modifier2.m**: Use this file for testing the model "net_RENet_VGG_Modified2".

<!-- 4. Now you can run either test models individually or run MATLAB GUI App as described below. -->

<!-- ### Directory: classification-test-code

-   **test_code_RENet_VGG_Modifier1.m**: Use this file for testing the model "net_RENet_VGG_Modified1" on folder of images at once.

### Directory: MATLAB-GUI-app

-   Inside this directory there is **gui_classification.mlapp** file. In order to use MATLAB-GUI-app type following at command window.

```MATLAB
>> gui_classification
```

Now the GUI interface will open after some time, then it will allow use to load image, and classify it as COVID-19 or Non-COVID-19. -->

## Co-Author

Prof. Asifullah Khan,

Department of Computer and Information Sciences (DCIS),

Pakistan Institute of Engineering and Applied Sciences (PIEAS).

Email: asif@pieas.edu.pk

faculty.pieas.edu.pk/asifullah/

## How to cite / More information

Khan, Saddam Hussain, Anabia Sohail, Muhammad Mohsin Zafar, and Asifullah Khan. "Coronavirus Disease Analysis using Chest X-ray Images and a Novel Deep Convolutional Neural Network." (2020), 10.13140/Rg. 2.2. 35868.64646 April (2020): 1-31.
