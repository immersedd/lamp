
# MP Framework
## Overview
This repository contains the implementation of LAMP, a framework for predicting working memory consumption of workloads.


## Requirements
- PostgreSQL version 15.6
```bash
# Download link
wget https://ftp.postgresql.org/pub/source/v15.6/postgresql-15.6.tar.gz
```
- Python 3.8+  
- Recommended libraries:
  - numpy==1.22.0
  - torch==2.4.1
  - scipy==1.10.1
  - psycopg2-binary==2.9.10
  - scikit-learn==1.3.2
  
Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
  - Place workload data files (JSON format) in workloads/data/.
  - Use workloads/sampler.py to generate workloads.
  - Example provided: F_job_N5_W30_R0.json.

## File Descriptions
  - src/
    - binary_tree_conv.py
      - Core implementation of Tree Convolutional Neural Networks (TCNN). 
    - deepsets.py
      - Implements DeepSets aggregation for workloads.
    - tree_builder.py
      - Constructs binary tree structures from query execution plan descriptions.
    - tree_featurizer.py
      - Extracts structured features (node attributes, operators, subexpressions) from query trees.
    - featurize.py
      - End-to-end feature engineering pipeline.
    - conf.ini
      - Default configuration file, specifying hyperparameters, dataset paths, and runtime options.
    - training.py
      - training script for workload prediction models.
  - workloads/
    - sampler.py
      - Implements workload sampling logic. Can generate workloads from query sets for training/testing.
  - model_utils/
    - modules for model training and evaluation.
## Usage
1. Edit the paths and parameters at the top of the script:
   - `PROJECT_ROOT`: project root directory
   - `PY_SCRIPT`: training entry script (default `src/training.py`)
   - `LOG_DIR`: output directory for logs
   - `JSON_PATHS`: list of workload data files
   - `MODELS`: list of model names to train

2. Adjust batch sizes and concurrency limits (`BATCH_JOBS_LIMITS`) as needed.

3. Run the script:
   ```bash
   bash run_training.sh

### Data Acquisition
  - ./run_workloads.sh           → use conf.ini
  - or batch execution of triplets → provide workload, output, db_name

