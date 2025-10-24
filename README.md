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

## Dependencies

See `requirements.txt` for a complete list of dependencies.

