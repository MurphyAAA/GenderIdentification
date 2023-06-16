# GenderIdentification


# Introduction

This project is aiming to develop a classifier to distinguish the gender from 12-dimensional features.
Training set has xxx samples, XXX of them are male, the rest XXX are female.
Expect woking point provided ($\pi$ = 0.5
C<sub>fn</sub>=1 C<sub>fp</sub>=1 ) , another woking point are also into considered($\pi$ = 0,1 C<sub>fn</sub>=1 C<sub>fp</sub>=1)

# Feature

label = 1 female
label = 0 male
### Histogram and 2D scatter plots of dataset features - principal components



| 1st principle component | scatter of first two components |
|:-----------------------:|:-------------------------------:|
| ![](images/gau_1st.jpg) |   ![](images/gau_scatter.jpg)   |
|                         |    ![](images/gau_2ndPC.jpg)    |

### Histogram of dataset features - LDA direction


 - LDA shows a linear classifier may be able discriminate the classes 
### Histogram of dataset features - LDA direction
![](images/LDA.jpg)


 - Gaussian may not sufficient for dividing the gender according to the observation from its first principal component
 - LDA shows that a linear classifier may be able to discriminate the classes to some degree, but, regarding the features we observed in scatter plot, no linear models (eg,GMM ) will have better performance 

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

We adopt K- fold protol with K = 5. We measure performance in terms of minimum costs.
Since two working points are considered, we compute the minDCF of both working points and their average (C<sub>prim</sub>)


We will assess the actual DCF( actual C<sub>prim</sub>) and score calibration once we have selected the top-performing model

## Gaussian classifier

### MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) | minDCF( $\widetilde{\pi}$ = 0.1) | min C<sub>prim</sub> |
|:---:|:--------------------------------:|:---------------------------------|:---------------------|
|  -  |              0.144               | 0.364                            | 0.250                |
| 11  |            **0.136**             | **0.349**                        | **0.243**            |
| 10  |              0 189               | 0.422                            | 0.306                |
|  8  |              0.261               | 0.513                            | 0.387                |
|  6  |              0.282               | 0.588                            | 0.435                |

11 dimension looks have the best performance among them

### Tied MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) | minDCF( $\widetilde{\pi}$ = 0.1) | min C<sub>prim</sub> |
|:---:|:--------------------------------:|:---------------------------------|:---------------------|
|  -  |            **0.127**             | **0.360**                        | **0.244**            |
| 11  |              0.129               | 0.361                            | 0.245                |
| 10  |              0 187               | 0.420                            | 0.304                |
|  8  |              0.257               | 0.532                            | 0.395                |
|  6  |              0.278               | 0.607                            | 0.443                |


### Naive MVG classifier - minDCF(K-Fold) 
| PCA | minDCF( $\widetilde{\pi}$ = 0.5) | minDCF( $\widetilde{\pi}$ = 0.1) | min C<sub>prim</sub> |
|:---:|:--------------------------------:|:---------------------------------|:---------------------|
|  -  |              0.456               | 0.768                            | 0.612                |
| 11  |            **0.136**             | **0.358**                        | **0.247**            |
| 10  |              0.184               | 0.421                            | 0.02                 |
|  8  |              0.257               | 0.526                            | 0.391                |
|  6  |              0.277               | 0.607                            | 0.442                |
The Naive MVG shows the similar answer with MVG but this method reduce computational complexity

que
## Logistic Regression classifier
## Support Vector Machine
## Gaussian Mixture Models

## Calibration and fusion
# Experimental Evaluation
now we analyze a model performance on the evaluation set. We start from the selected model and then different choices will be analyzed as well

Only min and actual C<sub>prim</sub> costs

# Conclusion

