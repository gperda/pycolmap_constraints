import pyceres
class Options:
    def __init__(self, config):
        # General section
        self.input_folder = config.get("General", "input", fallback=None)
        self.output_folder = config.get("General", "output", fallback=None)
        self.priors_delimiter = config.get("General", "priors_delimiter", fallback=";")

        # Problem section
        self.bundle = config.getboolean("Problem", "bundle", fallback=None)
        self.priors_file = config.get("Problem", "priors_file", fallback=None)
        self.rigs_file = config.get("Problem", "rigs_file", fallback=None)
        self.rigs = None
        self.priors_rotations_from_rec = config.getboolean("Problem", "priors_rotations_from_reconstruction", fallback=False)

        priors_covariance_str = config.get("Problem", "priors_covariance", fallback="[1,1,1,1,1,1]")
        if priors_covariance_str == "[]":  
            priors_covariance_str = "[1,1,1,1,1,1]"
        self.priors_covariance = self._parse_list(priors_covariance_str, float)

        self.fix3D_points = config.getboolean("Problem", "fix3D_points", fallback=False)
        self.use_priors_rotations = config.getboolean("Problem", "use_priors_rotations", fallback=False)

        fixed_cameras_ids_str = config.get("Problem", "fixed_cameras_ids", fallback="[]")
        self.fixed_cameras_ids = self._parse_list(fixed_cameras_ids_str)

        self.transform_reconstruction = config.getboolean("Problem", "transform_reconstruction", fallback=False)
        # Solver section
        solver_options = pyceres.SolverOptions()
        linear_solver_type = config.get("Solver", "linear_solver_type", fallback="SPARSE_SCHUR")
        linear_solver_type_mapping = {
            "SPARSE_SCHUR": pyceres.LinearSolverType.SPARSE_SCHUR,
            "DENSE_QR": pyceres.LinearSolverType.DENSE_QR,
            "None": pyceres.LinearSolverType.SPARSE_SCHUR,
        }
        solver_options.linear_solver_type = linear_solver_type_mapping.get(linear_solver_type, pyceres.LinearSolverType.SPARSE_SCHUR)
        solver_options.minimizer_progress_to_stdout = config.getboolean("Solver", "minimizer_progress_to_stdout", fallback=False)
        solver_options.num_threads = config.getint("Solver", "num_threads", fallback=4)
        self.solver_options = solver_options

    def _parse_list(self, list_str, value_type=int):
        try:
            return [value_type(x.strip()) for x in list_str.strip("[]").split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list format: {list_str}")
