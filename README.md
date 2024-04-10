# Climate State Classifier

Software to train/evaluate models to classify labeled climate data based on a convolutional neural network (CNN).

## Dependencies
- pytorch>=1.11.0
- tqdm>=4.64.0
- torchvision>=0.12.0
- torchmetrics>=0.11.2
- numpy>=1.21.6
- matplotlib>=3.5.1
- tensorboardX>=2.5
- tensorboard>=2.9.0
- xarray>=2022.3.0
- dask>=2022.7.0
- netcdf4>=1.5.8
- setuptools==59.5.0
- xesmf>=0.6.2
- cartopy>=0.20.2
- numba>=0.55.1

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate climclass
```

`environment-cuda.yml` should be used when working with GPUs using CUDA.

## Installation

`climclass` can be installed using `pip` in the current directory:
```bash
pip install .
```

## Usage

The software can be used to:
- train a model (**training**)
- make predictions using a trained model (**evaluation**)

### Input data
The input data samples must be given in the following naming convention, containing a single sample per file:
<category_name><sample_name><data_type><class_label>.nc

### Execution

Once installed, the package can be used as:
- a command line interface (CLI):
  - training:
  ```bash
  climclass-train
  ```
  - evaluation:
  ```bash
  climclass-evaluate
  ```
- a Python library:
  - training:
  ```python
  from climatestateclassifier import train
  train()
  ```
  - evaluation:
  ```python
  from climatestateclassifier import evaluate
  evaluate()
  ```

## Example

An example application can be found in the directory `demo`.
The instructions to run the example are given in the demo/README.md file.

## License

`Climate State Classifier` is licensed under the terms of the BSD 3-Clause license.

## Contributions

`Climate State Classifier` is maintained by the Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum).
- Current contributing authors: Johannes Meuer, Maximilian Witte, Étienne Plésiat, Christopher Kadow.
