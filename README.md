# Capstone Project: Betta Fish Evolutionary Morphology

2020 Fall

By Shijie He, Congcheng Yan, Chengchao Jin, Wancheng Chen, Chutian Chen

Mentor: Andres Bendesky

## Project Introduction
Betta fish is considered as a great candidate species for the domestic animal model to figure out how evolution can affect the traits of the domestic animals for the sake of its relatively short generation time and absence of ethical concern. In this project, we present a complete set of methods to automatically process Betta fish pictures in large quantities by using self-designed algorithms to calibrate and crop Betta fish images and transfer learning to segment fish bodies. Then we run the genetic mapping between genotypes and phenotypes to link genetic loci to individual traits which enable us to obtain convincing results to figure out which marker the features depend on and what type a gene is.

## Concept Slide
![image](https://github.com/cct15/2020_capstone/blob/main/concept.png)

## Color Calibration

## Fish Segmentation
The fish segmentation has two parts, whole fish cropping and fish anatomy. The whole fish cropping is to crop the fish from the images. Then we train a model with the idea of transfer learning to identify different parts of the body. 

[Whole_Fish_Cropping](https://github.com/cct15/2020_capstone/blob/main/fish_segmentation/whole_fish_cropping.ipynb)

The fish anatomy model is trained on FloydHub, inspired by: https://github.com/WillBrennan/SemanticSegmentation

## Feature Extaction
To separate different parts of the fish, feature extraction needs to be done to get the final data frame that is going to be used for the genetic mapping in the next step. Size and color distribution of the parts are two main features we need to extract from the fish pictures. 

[Feature Extraction](https://github.com/cct15/2020_capstone/tree/main/feature_extraction)

## Genetic Mapping
Quantitative trait locus (QTL) mapping is used in this section. QTL is a locus (section of DNA) that correlates with variation of a quantitative trait in the phenotype of a population of organisms. QTLs are mapped by identifying which molecular markers correlate with an observed trait. In this part, we will use both the data of phenotype, which is the feature matrix we have from previous work, and the genotype, which are given. 

[Genetic Mapping](https://github.com/cct15/2020_capstone/blob/main/genetic_mapping/qtl.Rmd)

## Conclusion and Future Work
In conclusion, to locate genomic regions affecting the size and color of the fish, first we develop models to read, calibrate, segment, analyze the fish images automatically. Then based on the features extracted from images and the genetic data, we run the genetic mapping between genotypes and phenotypes. From the results of genetic mapping, we could find which marker the features depend on. We could also find what type a gene is (dominant, semi-dominant or recessive).

The results are great overall. However, there are still some problems and need to be solved in future work.

The first problem is the poor performance of anatomy on fish with huge fins and tails. As figure 5.1 shows, the neural network can’t properly segment this kind of fish. Huge fins and tails are much more difficult to detect as they’re often overlapped. We can fine tune the neural network and expand the training set with this kind of fish to improve the performance.

The second problem is that it’s hard to detect some parts of the fish based on the side view. From the figure 5.2, we can see that the pectoral fin can’t be segmented from the side view. What’s more, overlap and curling of fins and tails lead to inaccurate features, especially the size. To solve these two problems, we could analyze the images from different angles (side view and top view) together.
