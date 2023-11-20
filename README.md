# Image Feature Extraction and Matching

## Overview

This repository contains code for extracting unique features from a set of images using the Harris operator and λ-operator, **all implemented from scratch** . Additionally, it implements feature descriptor generation using Scale Invariant Features (SIFT) and matches image set features using Sum of Squared Differences (SSD) and Normalized Cross-Correlations (NCC).

## A) Feature Extraction with Harris and λ-Operator

### Harris Operator

The Harris corner detector is employed to extract unique features in the grayscale and color images. The computation times for generating these points are reported.

### λ-Operator

A variant of the Harris operator, the λ-operator is utilized for feature extraction. Computation times for generating these points, considering the additional square root calculations, are reported.

## B) Feature Descriptors with SIFT

Scale Invariant Feature Transform (SIFT) is implemented from scratch to generate feature descriptors for the extracted points. The computation time for generating these descriptors is reported.

## C) Image Set Feature Matching

### SSD Method

Feature matching is performed using Sum of Squared Differences (SSD) from scratch. The computation time for this matching process is reported.

### NCC Method

Feature matching using Normalized Cross-Correlation (NCC) is implemented from scratch. The computation time is reported for this matching technique.


---

Feel free to explore the code for a detailed understanding of the implementation. For comprehensive insights and results, refer to the corresponding sections in the  [Computer Vision Report](<https://github.com/alaayasser01/Image-Feature-Extraction-and-Matching/blob/main/Computer%20vision%20report.docx>). available in the repository.
