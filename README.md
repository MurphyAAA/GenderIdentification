# GenderIdentification


# Introduction

This project is aiming to develop a classifier to distinguish the gender from 12-dimensional features.
Training set has 2400 samples, 720 of them are male, the rest 1680 are female.
According to the requirement, only one working point applied ($\pi$ = 0.5
C<sub>fn</sub>=1 C<sub>fp</sub>=1 ) [ 1 female   0 male]

# Feature


### Histogram and 2D scatter plots of dataset features - principal components



| 1st principle component | scatter of first two components |
|:-----------------------:|:-------------------------------:|
| ![](images/gau_1st.jpg) |   ![](images/gau_scatter.jpg)   |
|                         |    ![](images/gau_2ndPC.jpg)    |



### Histogram of dataset features - LDA direction
![](images/LDA.jpg)
 - Gaussian may not sufficient for dividing the gender according to the observation from its first principal component
 - LDA shows that a linear classifier may be able to discriminate the classes. But, regarding the features we observed in scatter plot, no linear models (eg,GMM ) maybe will have better performance 

### Pearson correlation coefficient for the dataset features

|          Dataset          |             Male              | Female                          |
|:-------------------------:|:-----------------------------:|:--------------------------------|
| ![](images/heatmap_D.png) | ![](images/heatmap_Dmale.png) | ![](images/heatmap_Dfemale.png) |

Dark color implies larger value means high correlation between those two features.
From the graph, we can found that no matter each gender, some features are significantly have large correlation with others.
It means we may benefit from using PCA to map data to xx, But an explained variance will also draw below to pick a suitable left dimension number
5
### PCA_explained_variance

![](images/PCA_explained_var.jpg)

with 10 dimension we could explain about 99% of the dataset variance. 97% with 8 directions 91% with 6 directions. To start, we will consider these three values for PCA.



# Building a classifier for the task

We adopt K- fold protol with K = 5. We measure performance in terms of minimum costs( minDCF).
Then, We will assess the actual DCF( actual C<sub>prim</sub>) and score calibration once we have selected the top-performing model

## Gaussian classifier
We test all three approaches(MVG, Naive Bayes model, Tied) also with the effect of PCA

### MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) | 
|:---:|:--------------------------------:|
|  -  |              0.144               | 
| 11  |            **0.136**             |
| 10  |              0 189               | 
|  8  |              0.261               | 
|  6  |              0.282               | 

When we reduce dimension into 11, the model looks have the best performance among them

### Tied MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) | 
|:---:|:--------------------------------:|
|  -  |            **0.127**             |
| 11  |              0.129               | 
| 10  |              0 187               | 
|  8  |              0.257               | 
|  6  |              0.278               | 

It shows the performance is better when dimension is in original size and when reduce 1, it increases a little but is still in tolerance. However, when dimension comes to 10, the cost increase significantly.


### Naive MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) |
|:---:|:--------------------------------:|
|  -  |              0.456               |
| 11  |            **0.136**             | 
| 10  |              0.184               | 
|  8  |              0.257               |
|  6  |              0.277               | 



The Naive MVG shows the similar answer with MVG but this method reduce computational complexity.

## Logistic Regression classifier

We now consider Logistic Regression models. 
We start analyzing the linear classifier without PCA

### Logic Regression classifier - minDCF(K-Fold) 

![](images/Linear_LR.jpg)
It is surprise that the performance is not quite worse since minDCF can reach 0.118, it is better than MVG model

Then we could apply PCA to see whether it will improve our model .
Because the z-norm doesn't favor our model's performance, we will apply PCA on original data 

It can be seen that the result show worse result when we reduce the dimension, when it jumps to 10 , the cost increase dramatically.
![](images/LR_Compare.jpg)
Up to know, the best model is linear Regression with original PCA.

## Support Vector Machine
We move to SVM model, we try linear SVM firstly

Then we try the kernel SVM, we start from polynomial kernels. Now we only consider original dimension (no PCA)

For RBF kernel

Because in some case, better performance can be seen when dimension reduce into 11, so we try to apply our xxx model with PCA data
## Gaussian Mixture Models
Finally, we explore another approaches, which is GMM classifiers.
## Calibration and fusion
We use the DET plot to compare the best models that we collect from now.
It can be seen that....
# Experimental Evaluation
now we analyze a model performance on the evaluation set. We start from the selected model and then different choices will be analyzed as well

Only min and actual C<sub>prim</sub> costs

# Conclusion

