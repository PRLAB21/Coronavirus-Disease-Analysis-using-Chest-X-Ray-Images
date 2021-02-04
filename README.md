# Coronavirus Disease Analysis using Chest X-Ray Images and a Novel Deep Convolutional Neural Network

The novel coronavirus (COVID-19) is quickly spreading throughout the world, but facilities in the hospitals are limited. Therefore, diagnostic tests are required to timely identify COVID-19 infected patients, and thus reduce the spread of COVID-19. The proposed method exploits the learning capability of the convolutional neural network (CNN) to classify COVID-19 infected versus healthy patients. The classification is accomplished using a new CNN architecture suitable for pneumonia-based analysis of COVID-19 chest X-ray images. The proposed COVID-19 RENet is an encoder-based CNN architecture that is well suited for feature extraction and image analysis. It is observed that the systematic dimensionality reduction through several layers combined with the synchronization of max-pooling (edge-based information extraction) and average pooling (Region-based information extraction) is well suited for image analysis. Finally, the deep features are extracted from CNN architecture and fed into the SVM classifier to improve the classification performance. The proposed technique is evaluated and compared with existing techniques using 5-fold cross-validation on the COVID-19 X-ray dataset. The proposed technique shows good performance and in most of the cases, outperforms the current techniques using metrics such as the accuracy, F-score, and ROC curve. The proposed approach (concatenated deep features of both the COVID-RENet and COV-VGGNet model) achieved the highest classification performance on COVID-19 X-ray images. Objective evaluation of proposed approach achieved an accuracy of 98.3%, AUC: 0.98, F-score: 0.98, Recall: 0.97, and Precision: 0.99, respectively.

In this repository, we provide the MATLAB GUI and Testing Code for the Coronavirus Disease Analysis using Chest X-ray Images for the research community to use our research work.

## Models Architecture

### Architectural details of the proposed COVID-RENet-1

![Architectural details of the proposed COVID-RENet-1](./repo-images/architecture-covid-renet-1.png "Architectural details of the proposed COVID-RENet-1")

### Architectural details of the proposed COVID-RENet-2

![Architectural details of the proposed COVID-RENet-2](./repo-images/architecture-covid-renet-2.png "Architectural details of the proposed COVID-RENet-2")

### Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification system

![Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification system](./repo-images/architecture-covid-19-classification-system.png "Proposed deep concatenated feature space (DCFS-MLC)-based COVID-19 classification system")

**Trained Model is available at [COVID-RENet-1](https://drive.google.com/file/d/1IY8Di0Jqlmb7pjw6OmKdmc2QasnLQ3sA/view?usp=sharing) and [COVID-RENet-2](https://drive.google.com/file/d/1ctjUFQLtNgMcKbQCYdPaPEsWXiBqhujM/view?usp=sharing) links.**

## Dataset

### Dataset Samples

Panel (A) and (B) show COVID-19 infected and healthy images, respectively.

![Dataset Samples](./repo-images/dataset-01.png "Dataset Samples")

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
