# Bias Mitigation in Language Models using a Multi-Dimensional Evaluation and Comparison Framework

A comprehensive framework for detecting, measuring, and mitigating bias in Large Language Models (LLMs). This project provides tools and techniques to evaluate various forms of bias and implement mitigation strategies to improve fairness in LLM outputs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation)
- [Datasets Used](#datasets-used)
- [Supported Bias Types](#supported-bias-types)
- [Mitigation Techniques](#mitigation-techniques)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

This project addresses the critical challenge of bias in Large Language Models by providing a systematic approach to:

- **Detect** various forms of bias in LLM outputs
- **Measure** bias using comprehensive evaluation metrics
- **Mitigate** bias through state-of-the-art techniques
- **Analyze** the effectiveness of mitigation strategies

The framework supports multiple bias types including gender, racial, religious, and socioeconomic bias, with extensible architecture for adding new bias categories and mitigation methods.

## Features

- **Comprehensive Bias Detection**: Multi-dimensional bias evaluation across different demographic groups
- **Robust Metrics**: Statistical and semantic metrics for bias quantification
- **Multiple Mitigation Techniques**: Pre-processing, in-processing, and post-processing approaches
- **Detailed Analysis**: Visualization and reporting of bias patterns and mitigation effectiveness
- **Modular Architecture**: Easy to extend with new datasets, metrics, and mitigation methods
- **Reproducible Results**: Standardized evaluation protocols and result tracking

## Project Structure

```
bias-mitigation-llm/
├── processed_datasets/          # Processed CSV datasets
├── results/                     # Experimental results and outputs
├── src/
│   ├── data_loader.py          # Data loading and preprocessing utilities
│   ├── models.py               # LLM model loading 
│   ├── utils.py                # General utility functions
│   ├── mitigation/             # Bias mitigation techniques
│   └── metric/                 # Bias evaluation metrics
├── analysis.py                 # Main analysis script
├── metric_main.py              # Bias evaluation pipeline
├── mitigation_main.py          # Bias mitigation pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/heetabhanushali/Bias-Mitigation-in-LLMs.git
   cd Bias-Mitigation-in-LLMs
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run python files in the following order:**
   ```bash
   python3 src/data_loader.py   #to load and save the datasets
   python3 metric_main.py       #to measure bias using various metric
   python3 mitigation_main.py   #to implement various mitigation techniques
   python3 analysis.py          #to get results and plots
   ```

## Datasets Used



## Supported Bias Types

- **Gender Bias**: Stereotypes and discrimination based on gender
- **Racial Bias**: Prejudice related to race and ethnicity
- **Religious Bias**: Discrimination based on religious beliefs
- **Age Bias**: Stereotypes related to age groups
- **Socioeconomic Bias**: Prejudice based on economic status
- **Disability Bias**: Discrimination against people with disabilities


## Mitigation Techniques

### Context Injection
- Injects neutralizing context before the prompt to balance biased assumptions.

### Post-Filtering
- Generates multiple completions per prompt and selects the least toxic one using Detoxify scores.

### Prompt Cleaning
- Removes subtly or explicitly toxic phrasing before model generation using rule-based filters, lexical matching, and heuristics.

### Prompt Engineering
- Replaces biased words/phrases with neutral words/phrases before model generation.


## Evaluation Metrics

## WEAT (Word Embedding Association Test)
- Measures association between identity terms and attributes (like pleasant/unpleasant) using word embeddings

## CAT (Cloze-style Association Test)
- Cloze-style fill-in-the-blank test for detecting implicit bias in completions

## Toxicity Detection
- Uses Detoxify or Perspective API to measure offensive or toxic language

## Custom IBS (Intrinsic Bias Score)
- Intrinsic Bias Score combining contextual and identity-aligned embeddings to score intersectional bias


## Citation

If you use this project in your research, please cite:

```bibtex

```

## Contact

For questions, issues, or suggestions, please:
- Open an issue on the GitHub
- Contact the maintainers at [heetabhanushali@gmail.com]

---

**Note**: This project is for research and educational purposes. Please ensure compliance with ethical guidelines and regulations when working with sensitive data and bias evaluation.
