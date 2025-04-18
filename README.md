# pycolmap_constraints

## Overview
`pycolmap_constraints` is a Python script that leverages `pycolmap` and `pyceres` to optimize 3D reconstructions using pose priors and bundle adjustment.

## Usage

### Running the Script
To run the `main.py` script, use the following command:
```bash
python main.py --config /path/to/config.ini
```

### Configuration File (`config.ini`)
The script requires a configuration file in `.ini` format to specify input, output, and optimization parameters. Below is a description of the sections and parameters:

#### `[General]`
- `input`: Path to the input folder containing the reconstruction data.
- `output`: Path to the output folder where the optimized reconstruction will be saved.
- `priors_delimiter`: Delimiter used in the priors file (e.g., `;`).

#### `[Problem]`
- `bundle`: Boolean (`true` or `false`) to enable or disable bundle adjustment.
- `priors`: Path to the file containing pose priors.
- `priors_covariance`: Covariance matrix for the priors, specified as a list (e.g., `[0.1, 0.1, 0.1, 100, 100, 100]`).
- `use_priors_rotations`: Boolean to indicate whether to use rotation priors.
- `priors_rotations_from_rec`: Boolean to use rotations from the reconstruction instead of the priors file.
- `transform_reconstruction`: Boolean to apply an affine transformation to align the reconstruction with the priors.
- `fixed_cameras_ids`: List of camera IDs to keep fixed during optimization (e.g., `[1, 2, 3]`).
- `fix_3D_points`: Boolean to fix 3D points during optimization.

#### `[Solver]`
- `linear_solver_type`: Type of linear solver to use (e.g., `SPARSE_SCHUR`, `DENSE_QR`).
- `minimizer_progress_to_stdout`: Boolean to print solver progress to the console.
- `num_threads`: Number of threads to use for optimization.

### Example Configuration File
Below is an example `config.ini` file:
```ini
[General]
input = /path/to/input
output = /path/to/output
priors_delimiter = ;

[Problem]
bundle = true
priors = /path/to/priors.txt
priors_covariance = [0.1, 0.1, 0.1, 100, 100, 100]
use_priors_rotations = true
priors_rotations_from_rec = false
transform_reconstruction = true
fixed_cameras_ids = [1, 2, 3]
fix_3D_points = false

[Solver]
linear_solver_type = SPARSE_SCHUR
minimizer_progress_to_stdout = true
num_threads = 4
```

### Output
The optimized reconstruction will be saved in the specified output folder. Additional files such as `optimized.txt` and `cloud_optimized.txt` will contain the optimized camera positions and 3D points, respectively.

## Dependencies
- `pycolmap`
- `pyceres`
- `numpy`
- `configparser`
- `argparse`