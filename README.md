# Optimized Forest Thinning

## Overview
This project develops and evaluates systematic forest thinning strategies to optimize stand management by maximizing retention of high-value trees while removing small trees. The work implements two-phase thinning: initial thinning uses systematic k-row patterns (3/4/5-row) or optimized variable-row selection to establish corridors and reduce density by 25-33%, while secondary thinning applies targeted strategies based on thinning requirements. 
The framework provides comprehensive metrics, visualization tools, and interactive dashboards, all implemented as reusable Python libraries for decision support in operational forest management.

## Project Structure
```
├── data/                          # Processed data files
├── Raw-datasets/                  # Original, immutable raw datasets
├── Adaptive_User_Interface.ipynb  # [Description]
├── Artificial_Stand.ipynb         # [Description]
├── Combined_thinning_strategies.ipynb  # [Description]
├── Data_cleaning.ipynb            # Data preprocessing and cleaning
├── data_exploration.ipynb         # Exploratory data analysis
├── Diliwyn-XY.ipynb              # [Description]
├── Ranking-methods.ipynb          # [Description]
├── row_thinning_optimization.ipynb # [Description]
├── Section_Thinning.ipynb         # [Description]
├── Spatial_map.ipynb              # [Description]
├── Two_Stages_Ext.ipynb           # [Description]
├── Two_Stages.ipynb               # [Description]
├── Variable_thinning.ipynb        # [Description]
├── clean-thinning_data-2.csv      # Cleaned dataset
├── initial_thinning_lib.py        # Core library functions
├── secondary_thinning_lib.py      # Additional library functions
└── requirements.txt               # Python package dependencies
```

## Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone https://github.com/amith2610/Optimized-Thinning-Analysis.git
cd Optimized-Thinning-Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv thinning_env
source thinning_env/bin/activate  # On Windows: thinning_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the desired notebook from the list above

3. Run the cells sequentially

## Data

- **Raw-datasets/**: Contains the original raw data files
- **data/**: Contains processed and intermediate data files

## Forest Stands
- ### Dillwyn Stand (Stand-A):
  Trees: 2,403 total
  <img width="640" height="505" alt="Screenshot 2025-10-24 at 1 43 57 PM" src="https://github.com/user-attachments/assets/e521d183-b986-4f6f-adcc-0a6c44cdf0b8" />

- ### Suffolk Stand (Stand B):
  Trees:2,595 total
  <img width="630" height="407" alt="Screenshot 2025-10-24 at 1 43 23 PM" src="https://github.com/user-attachments/assets/7b96b9df-ae03-4a02-a904-11e8460d98dd" />

- ### Appomattox Stand (Stand C):
  Trees: 3,420 total
  <img width="595" height="558" alt="Screenshot 2025-10-24 at 1 42 44 PM" src="https://github.com/user-attachments/assets/0315e6c7-49f6-477e-85e1-b94eea410e54" />

  

## Dependencies

See `requirements.txt` for a complete list of dependencies.

