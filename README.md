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

## Feature Extaction
To separate different parts of the fish, feature extraction needs to be done to get the final data frame that is going to be used for the genetic mapping in the next step. Size and color distribution of the parts are two main features we need to extract from the fish pictures. 

[Feature Extraction](https://github.com/cct15/2020_capstone/blob/main/Feature_Extraction.ipynb)

## Genetic Mapping
Quantitative trait locus (QTL) mapping is used in this section. QTL is a locus (section of DNA) that correlates with variation of a quantitative trait in the phenotype of a population of organisms.[7] QTLs are mapped by identifying which molecular markers correlate with an observed trait. In this part, we will use both the data of phenotype, which is the feature matrix we have from previous work, and the genotype, which are given. 

[Genetic Mapping](https://github.com/cct15/2020_capstone/blob/main/qtl.Rmd)
