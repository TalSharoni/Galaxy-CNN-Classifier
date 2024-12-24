# CNN-Based Galaxy Classification: Final Project for PHYS-512

### Introduction  
This project focuses on classifying galaxies into three categories: Spiral, Elliptical, and Irregular using Convolutional Neural Networks. CNNs are a powerful deep learning architecture widely used for image classification tasks due to their ability to learn and generalize feature representations directly from raw pixel data.

For this project, I implemented two models:  
1. **A Keras-based CNN** using pre-existing layers from TensorFlow Keras. I configured the layer ordering, architecture, and hyperparameters to achieve the best possible accuracy. 
2. **A CNN from scratch**, where I manually implemented each layer using nothing more than Numpy operations and for loops, to better understand the mechanics and math behind CNNs.

### Motivation
It is estimated that there are between 200 billion to 2 trillion galaxies in the observable universe. We have images such as the Hubble Legacy Field containing an estimated 265,000 galaxies in a single frame, and with novel telescopes such as JWST and future successors we will be able to capture more images of galaxies, at higher resolutions, and with greater depth than ever before.

While this is incredibly exciting, manually labeling these images and searching for patterns would be slow, could produce human error, and inefficient. Accurately classifying these galaxies is essential for understanding their formation and evolution and so having a computer doing this task successfully on huge datasets is an important goal.

As of personal motivation, I'm deeply interested in both astrophysics, early universe and galaxies evolution and machine learning, especially computer vision so I was excited to tackle a project like this, especially using real world data.

### Dataset
There were two datasets I used for the project, both of which are too large to include in the GitHub repository:
1. **[Galaxy Zoo 2: Images](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images)**  
   - This dataset contains images of individual galaxies.  
   - It also includes a mapping file that associates each image with its corresponding Galaxy Zoo objid.

2. **[Galaxy Zoo Catalog](https://data.galaxyzoo.org/)**  
   - This dataset provides galaxy classifications based on human votes. I specifically used Table 2, from which I used the debiased columns for the labeling.
  
From the dataset, I extracted 2000 images, which I randomly flipped in four directions to improve the model’s generalizability.

If the debiased probability of spiral (P_CS_DEBIASED) or elliptical (P_EL_DEBIASED) was over 0.65 it was labeled as such, otherwise, it was labeled as Irregular. This yielded the most (relatively) balanced dataset.

I also zoomed in on the images. Initially, the images were of size (424, 424), but I zoomed in without changing the resolution to (224, 224). This adjustment helped remove empty space or noise from other galaxies, which was beneficial for training.
   
### Project Files
The files for the project pipeline are as follow:

1. **`src/extract.py`**  
   - This script is responsible for extracting and organizing the original dataset. It extracts 2000 galaxies at random, classifies them as explained above and stores the clean data as well as images in a csv. 

2. **`src/preprocess.py`**  
   - This script handles the preprocessing of the data, zooming in, normalizes the pixel values by dividing by 255.0, and stores the final processed data in a pickle file.

3. **`galaxies_model.ipynb`**  
   - This Jupyter notebook contains the entire workflow for building and training the models. It includes the implementation of both the Keras CNN and the custom CNN built from scratch. It loads the preprocessed dataset, builds the models and evaluate the performance of the models. For the Keras CNN, I included an evolution plot showing how the model's loss and accuracy change over epochs. It also includes a confusion matrix and a sample of 9 wrongly labeled galaxies to inspect where the model might be failing.

### Results
The results of the Keras CNN are promising, especially given the messy data, with an accuracy in the 70% range. It also shows that it labels both spiral and elliptical with solid accuracies in the 80s or higher, but is struggling more with the irregular galaxies. This makes sense since using irregular as a catch-all for low confidence labels might hurt generalizability. Additionally, there are some flaws in the classification, such as consistently labeling blue galaxies as Spiral, which is not always correct.

The results of the model built from scratch are less promising. The model doesn’t appear to be improving across epochs, which may be due to the compromises made to speed up the implementation (as it was relatively slow) and potential issues with gradient calculation or backpropagation. Despite these challenges, implementing the layers from scratch provided valuable insights into the inner workings of CNNs, especially for someone relatively new to the field.
