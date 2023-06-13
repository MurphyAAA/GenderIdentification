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
| ![](images\gau_1st.jpg) |   ![](images\gau_scatter.jpg)   |
|                         |    ![](images\gau_2ndPC.jpg)    |

### Histogram of dataset features - LDA direction


 - LDA shows a linear classifier may be able discriminate the classes ??

 - Male and female are characterized by multiple clusters
 - Maybe be GMM will model well for both of our data
 - Gaussian may not sufficient for dividing the gender according to the observation from its first principal component

### Pearson correlation coefficient for the dataset features

|          Dataset          |              Male               | Female |
|:-------------------------:|:-------------------------------:|:-------|
| ![](images\heatmap_D.png) | ![](images\heatmap_Dmale.png) |![](images\heatmap_Dfemale.png)|

Dark color implies larger value means high correlation between those two features.
From the graph, we can found that no matter each gender, some features are significantly have large correlation with others.
It means we may benefit from using PCA to map data to xx, But an explained variance will also draw below to pick a suitable left dimension number
5
### PCA_explained_variance

![](images\PCA_explained_var.jpg)

with 10 dimension we could explain about 99% of the dataset variance. 97% with 8 directions 91% with 6 directions. To start, we will consider these three values for PCA.



# Building a classifier for the task

We adopt K- fold protol with K = 5. We measure performance in terms of minimum costs.
Since two working points are considered, we compute the minDCF of both working points and their average (C<sub>prim</sub>)


We will assess the actual DCF( actual C<sub>prim</sub>) and score calibration once we have selected the top-performing model

## Gaussian classifier
## Logistic Regression classifier
## Support Vector Machine
## Gaussian Mixture Models

## Calibration and fusion
# Experimental Evaluation
now we analyze a model performance on the evaluation set. We start from the selected model and then different choices will be analyzed as well

Only min and actual C<sub>prim</sub> costs

# Conclusion

