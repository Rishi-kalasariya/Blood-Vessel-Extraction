# Blood-Vessel-Extraction

## Overview

This project extracts the entire blood vessel structure from a given **fundus image** and presents it as a **binary map**. Unlike conventional approaches that rely on machine learning, this method is built entirely **from scratch** using image processing techniques, achieving **remarkable accuracy**.

### 📝 Files Included
- **`Extraction.py`** – Contains the core implementation for vessel extraction.
- **`colab_file.ipynb`** – A Colab notebook demonstrating the input/output process.
- **Both files are well-commented** for better understanding.

### 🔥 Key Highlights
- **No Machine Learning:** The entire process is built from **scratch** using image processing techniques.
- **High-Quality Output:** Despite not using ML, the method provides **excellent results**.
- **Dice Score:** Achieves a **Dice score of 0.7** when compared to ground truth.

## Features

- Thresholding: Converts an image to a binary format.
- Connected Components Analysis: Identifies and labels connected regions in a binary image.
- CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances the contrast of an image adaptively.

## Input & Output

- Input: Grayscale or color images in a supported format.
- Output: Processed images saved to a designated folder.
