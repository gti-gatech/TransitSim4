# TransitSim 4.0

Welcome to TransitSim 4.0, a powerful tool for transit modeling and analysis. This repository contains the necessary scripts, documentation, and case studies to help you get started with TransitSim 4.0.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Case Studies](#case-studies)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
TransitSim 4.0 is a Python-based tool designed for transit modeling and analysis. It leverages the RAPTOR algorithm to provide efficient transit routing and analysis capabilities. This repository includes detailed documentation, example case studies, and all necessary scripts to run TransitSim.

---

## Installation
To install TransitSim 4.0, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TransitSim4.git
   cd TransitSim4
   ```
2. Set up a virtual environment (recommended):
   ```bash
   python -m venv transitsim_env
   source transitsim_env/bin/activate  # On Windows use `transitsim_env\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
To get started with TransitSim, open the ```transitsim.ipynb``` Jupyter notebook. This notebook provides a step-by-step guide on how to use TransitSim, including data preprocessing, running the RAPTOR algorithm, and post-analysis.

---

## Case Studies
This repository includes two case studies:
* **Heat Exposure Analysis**: See ```project_heat.ipynb``` for a detailed walkthrough.
* **Food Accessibility Analysis**: See ```project_food.ipynb``` for a detailed walkthrough.

---

## Folder Structure
Here’s an overview of the repository structure:

```
TransitSim4/
├── README.md
├── transitsim.ipynb
├── project_heat.ipynb
├── project_heat/
├── project_food.ipynb
├── project_food/
├── docs/
├── program/
├── data/
│   ├── gtfs/
│   ├── transfer_network/
│   ├── output/
│   ├── postproc/
│   ├── trajectories/
│   ├── linkshp/
│   ├── linkpts/
│   ├── scratch/
│   ├── gtfs_output/
│   ├── transfer_output/
│   ├── dicts/
│   ├── sample_processed/
│   ├── f_l_path/
├── LICENSE
```

---

## License
