# ADS Final Project
* Project Member : Yen Le, Hannah Phung and Manjiri Bhandarwar
* Project Goal : Skin disease detection by images using Deep Learning 

# Project Description

* Skin disease classification is a task that aims to create a tool to assist dermatologists in making accurate skin disease diagnosis. The popular deep learning models for our problem are CNNs, RNNs and GANs. We refer to six related works in the field and decided to use pre-trained CNNs models with ImageNet to deal with skin disease classification tasks. We also aim to use diverse techniques to improve performances and evaluate their safety, reliability and generalizability. 

* The project evaluates the performance of existing DenseNet201, ResNet50, along with Average Ensemble, Hierarchical Ensemble, CNNs, Siamese Networks, Multimodal Classification on a series of metrics:

  * Variation of model’s loss and accuracy in both training and testing process.
  
  * Confusion Matrix Classification Report that includes accuracy, presion, recall and avengers f-1 score.
  
  * Model’s performance using complex ensemble and multimodal data.
  
  * Model’s ability to generalize to tasks in new datasets (ISIC-2019 dataset).


## Project Flow
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/method.png)

## Dataset Preprocessing

Dataset: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

* Handling Metadata:
  * Cleaned missing values (e.g. imputed null age values with mean)
  * Removed rows with age 0 or 'unknown' sex
  * Minor reduction in observations for specific classes
  * Categorically encoded lesion types as 'cell type idx' (target)
* Image Processing: resized to 125x100 pixels, normalized by dividing by 255 and standardization applied on training data
* Handling Data Imbalance: Avoided random oversampling due to huge class imbalance (potential overfitting issues)
* Data Split: Split dataset: 80-20 train-test, further split train into 90-10 train-validation
* Implemented data augmentation using ImageDataGenerator
  * Random rotation (0-10 degrees)
  * Horizontal and vertical shifts
  * Random zoom (factor 0.1)
  * Random flipping of images

## Project Framework / Model Architectures
**Baseline models:** Pretrained DenseNet201 and ResNet50, 95 Base layers freezed and fine-tuned with HAM10000 dataset. Also used learning rate scheduler: ReduceLRONPlateau + Exponential decay and early Stopping
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/res.png)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/dense.png)
Build on top of baseline model by adding other features to help with performance and avoid overfitting: Data Augmentation, Input Normalization, Batch Normalization, and Dropout

Constructing more complex ensemble models 

**Average Ensemble:** 
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/average_ensemble.png)

**Classes that are usually misclassified as each other:**
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/class12.png)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/class36.png)

**Hierarchical Ensemble:** 
- To deal with data imbalance, each classifier focuses on classes or combinations of classes that have comparable size. For example, classifier 1 distinguishes between class 0 (around 66% of the dataset) and the remaining 6 classes
- The final stage of the pipeline focuses on pairs of classes (class 1 vs 2 and class 3 vs 6)  that are usually mistaken for each other in the single model case. 
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/hierachical.png)

**Multimodal approach:** 
Multimodal approach incorporate metadata including age, sex, and localization (i.e. where on the body the disease appear)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/multimodal.png)

**Using the best model to test on secondary dataset:** 

The best model is multimodal with DenseNet201
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/secondary_data.png)

# Code Details
* Collab Notebooks are annotated with appropriate subheadings for each part. 

# Results and Observations

## Milestone 2

![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/download-5.png)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/download-6.png)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/download-7.png)

* For DenseNet201, as we progress from the baseline model to the model that uses both batch normalization and dropout, the accuracy stays around the same (0.78-0.80). For ResNet50, accuracy drops after adding input normalization. For DenseNet201, the macro-averaged F1 score increased up till and peaked after adding input normalization, and for ResNet50 it increased up till and peaked after adding data augmentation and then started decreasing for both. Batch normalization and dropout reduced F1 scores for both DenseNet201 and ResNet50, but way more significantly for ResNet50.
* Best Model from Milestone 2: DenseNet201 with Data Augmentation and Input Normalization (macro average F1 0.64, accuracy of 0.80). For minority class (6) F1 is 0.42.

## Milestone 3

![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/download-1.png)
![alt text](https://github.com/hannahphung/ADS_finalproject/blob/main/img/f1scorem3.png)

* Best Model from Milestone 3: Multimodal DenseNet201 (macro average F1 0.75, accuracy of 0.83)
* There is a significant improvement in the macro average F1 score. For minority class (6) F1 doubles 0.84 from previous milestone.

## Discussion/Insights
* Model performs well on majority classes but not on the minority class due to high class imbalance.
  
* Multi-modal model performs the best out of all since it gives additional context (age, sex, localization)  to our image data!
  
* Ensemble methods to outperform individual models due to their ability to combine multiple models to produce a stronger, more robust prediction.
  
* Data nature also matters along with sample size
  
* In our future work, we could take other approaches on image preprocessing to increase the distinctiveness of each class, such as adjusting saturation and contrast, or find a hair removal software like our initial plan, rather than simply enlarging data size!
  
* Sometimes the goodness of a model is random (we trained a model from scratch twice because once the checkpoint wasn’t saved, and we discovered that their results differ). 




# References
* Datasets:
  
  * HAM10000: (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    
  * ISIC-2019: (https://challenge.isic-archive.com/data/#2019)
    
* Models we evaluated:
  * DenseNet201: (https://www.kaggle.com/code/farshadjafari97/skincancer-detection-multiple-models-83-accuracy)
    
  * ResNet50: (https://www.kaggle.com/code/farshadjafari97/skincancer-detection-multiple-models-83-accuracy)
   
* Model for Data Augmentation:
  
    * ImageDataGenerator: (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
  



