# Machine Learning Prediction of Pediatric Adverse Drug Reactions Using Consensus-Derived Scarce Data
Project Overview

This project aims to study predictive models for adverse drug reactions in pediatrics, utilizing computational biology and machine learning techniques to enhance drug safety and efficacy. Our goal is to provide data-driven support for clinical practices to optimize pediatric medication.

<img width="552" height="414" alt="image" src="https://github.com/user-attachments/assets/f0dc4dac-da7c-4b85-b6e3-cb802eedc58a" />

# Pediatric Adverse Drug Reactions Dataset

This dataset accompanies the manuscript:

**"Machine Learning Prediction of Pediatric Adverse Drug Reactions Using Consensus-Derived Scarce Data"**  
*Communications Chemistry* - Accepted in principle

## Dataset Description

This dataset contains molecular structures (SMILES format) and their associated adverse drug reactions (ADRs) specifically curated for pediatric populations. The data was derived using consensus strategies to address the inherent scarcity of pediatric pharmacovigilance data.

## File Structure

`final_dataset.csv`: Main dataset containing SMILES representations and labeled ADR tasks

**Format:**
First column: SMILES notation of molecular structures
Remaining columns: Binary labels for individual ADR tasks (1 = positive, 0 = negative, empty = unlabeled)

## Biological Fingerprints

For the Biofeat used in our study, we utilized the **Chemical Checker** approach. Due to the large model size, we provide only the code for converting SMILES to biological fingerprints in this repository. For detailed implementation and pre-trained models, please refer to:

**https://github.com/ersilia-os/eos4u6p**

## Limitations

Users should be aware of the following limitations when utilizing this dataset:

1. **Data Scarcity**: Pediatric ADR data is inherently limited compared to adult populations, resulting in smaller sample sizes for certain ADRs.

2. **Consensus-Based Labeling**: The dataset employs consensus strategies to maximize data utility, which may introduce aggregation biases across different data sources.

3. **Reporting Bias**: As with all pharmacovigilance data, this dataset may be subject to underreporting, selective reporting, and other biases inherent in spontaneous adverse event reporting systems.

4. **Age Group Heterogeneity**: The pediatric population encompasses diverse developmental stages (neonates, infants, children, adolescents), which may not be uniformly represented.

5. **Structural Representation**: SMILES-based molecular representation may not fully capture stereochemistry and 3D structural features relevant to biological activity.

## Citation

If you use this dataset, please cite our paper.

