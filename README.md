# DNA Sequencing and Risk Analysis using Hidden Markov Models (HMM)

## Overview

This project involves generating DNA sequences and conducting risk analysis for hereditary diseases using Hidden Markov Models (HMM) in Python. The focus is on analyzing genetic sequences to assess the likelihood of hereditary diseases, with a specific emphasis on the BRCA1 gene, which is associated with breast cancer risk.

## Features

- **DNA Sequence Alignment:** Aligns input DNA sequences using FASTA format.
- **Feature Extraction:** Extracts relevant features from DNA sequences for analysis.
- **HMM Modeling:** Trains and applies a Hidden Markov Model to predict disease risk.
- **Risk Analysis:** Provides a risk assessment for hereditary diseases based on genetic data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sushan-Adhikari/DNA_Sequencing_and_HMM.git
   cd DNA_Sequencing_and_HMM


2. Install required dependencies:
```bash
python install_requirements.py

```
3. Usage
Data Collection: Place your DNA sequences in FASTA format inside the brca1_sequences.fasta file.
Run Pipeline: Execute the pipeline for sequence alignment, feature extraction, and risk analysis:

```bash
python run_pipeline.py

```
4. Files
- **data_collection.py:** Script to collect and align DNA sequences.
- **feature_extraction.py:** Extracts features from aligned sequences.
- **hmm_model.py:** Builds and applies the Hidden Markov Model.
- **risk_analysis.py:** Analyzes the risk of hereditary diseases.
- **requirements.txt:** Lists the required Python packages.

5. License
This project is licensed under the MIT License - see the LICENSE file for details.
