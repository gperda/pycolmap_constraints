import pyceres
import pycolmap
import numpy as np
import pycolmap.cost_functions
import utils
import configparser
import argparse

class Options:
    def __init__(self, config):
        # General section
        self.input_folder = config.get("General", "input", fallback=None)
        self.output_folder = config.get("General", "output", fallback=None)

        # Problem section
        self.bundle = config.getboolean("Problem", "bundle", fallback=None)
        self.priors_file = config.get("Problem", "priors", fallback=None)
        self.priors_delimiter = config.get("General", "priors_delimiter", fallback=";")
        self.priors_rotations_from_rec = config.getboolean("Problem", "priors_rotations_from_rec", fallback=False)
        priors_covariance_str = config.get("Problem", "priors_covariance", fallback="[1,1,1,1,1,1]")
        if priors_covariance_str == "[]":  
            priors_covariance_str = "[1,1,1,1,1,1]"
        self.priors_covariance = self._parse_list(priors_covariance_str, float)
        self.fix3D_points = config.getboolean("Problem", "fix3D_points", fallback=False)
        self.use_priors_rotations = config.getboolean("Problem", "use_priors_rotations", fallback=False)
        fixed_cameras_ids_str = config.get("Problem", "fixed_cameras_ids", fallback="[]")
        self.fixed_cameras_ids = self._parse_list(fixed_cameras_ids_str)
        self.transform_reconstruction = config.getboolean("Problem", "transform_reconstruction", fallback=False)
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

    # def validate(self):
    #     if not self.input_folder or not self.output_folder or self.bundle is None:
    #         raise ValueError("Missing required parameters in the configuration file.")

def read_reconstruction(folder_path):
    try:
        reconstruction = pycolmap.Reconstruction(folder_path)
        print(f"Successfully loaded reconstruction from {folder_path}")
        return reconstruction
    except Exception as e:
        print(f"Error reading reconstruction: {e}")
        return None

def get_pose_prior_from_line(line: str, delimiter: str):
    els = line.split(delimiter)
    if len(els) == 4:
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        rotation = None  # Rotation is not provided
    elif len(els) == 7:
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        rotation = np.array([float(els[4]), float(els[5]), float(els[6])])
    else:
        raise ValueError(f"Invalid number of parameters in line: {line}")
    return (image_name, position, rotation)

def read_priors(rec, options):
    with open(options.priors_file, "r") as file:
        priors = {}
        for line in file:
            if line[0] == "#":
                continue
            image_name, position, rotation = get_pose_prior_from_line(line, options.priors_delimiter)
            img_object = next((i for i in rec.images.values() if i.name == image_name), None)
            if img_object:
                if image_name not in priors:
                    priors[image_name] = {}
                priors[image_name]['position'] = position
                if options.use_priors_rotations:
                    if options.priors_rotations_from_rec:
                        priors[image_name]['rotation'] = img_object.cam_from_world.rotation
                    elif rotation is not None and rotation.any():
                        priors[image_name]['rotation'] = pycolmap.Rotation3d(rotation)
                else:
                    priors[image_name]['rotation'] = None
    file.close()
    options.priors = priors

def define_problem(rec, options):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()

    if options.priors_file:
        cov = np.diag(options.priors_covariance)
        first_image_position = np.array([0, 0, 0], dtype=np.float64)
        first_image_name = list(options.priors.keys())[0]
        if options.transform_reconstruction:
            first_image_position = options.priors[first_image_name]['position']
        for im in rec.images.values():
            if options.use_priors_rotations and options.priors[first_image_name]['rotation']:
                cost = pycolmap.cost_functions.AbsolutePosePriorCost(cov, pycolmap.Rigid3d(options.priors[im.name]['rotation'],-options.priors[im.name]['rotation'].matrix() @ (options.priors[im.name]['position']-first_image_position)))
            else:
                cost = pycolmap.cost_functions.AbsolutePosePositionPriorCost(cov[:3, :3], options.priors[im.name]['position']-first_image_position)
            pose = im.cam_from_world
            prob.add_residual_block(cost, loss, [pose.rotation.quat, pose.translation])
            prob.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())

    if options.bundle:
        for im in rec.images.values():
            cam = rec.cameras[im.camera_id]
            for p in im.points2D:
                if p.point3D_id in rec.points3D:
                    cost = pycolmap.cost_functions.ReprojErrorCost(cam.model, p.xy)
                    pose = im.cam_from_world
                    params = [
                        pose.rotation.quat,
                        pose.translation,
                        rec.points3D[p.point3D_id].xyz,
                        cam.params,
                    ]
                    prob.add_residual_block(cost, loss, params)
            prob.set_manifold(
                im.cam_from_world.rotation.quat, pyceres.EigenQuaternionManifold()
            )
    
    for cam_id in options.fixed_cameras_ids:
        prob.set_parameter_block_constant(rec.cameras[cam_id].params)
    if options.fix3D_points:
        for p in rec.points3D.values():
            prob.set_parameter_block_constant(p.xyz)
    
    return prob

def solve(prob, options):
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    solver_options = options.solver_options
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, prob, summary)
    print(summary.BriefReport())

def transform_reconstruction(rec, options):
    projection_centers_array = np.array([im.projection_center() for im in rec.images.values()]).T
    prior_positions_array = np.array([options.priors[im.name]['position'] for im in rec.images.values()]).T
    M = utils.affine_matrix_from_points(projection_centers_array,prior_positions_array, False, True, True)
    first_image_name = list(options.priors.keys())[0]
    first_image_position = options.priors[first_image_name]['position']
    M[:3, 3] = M[:3, 3] - first_image_position
    S = pycolmap.Sim3d(M[:3, :])
    rec.transform(S)
    return M

def write_reconstruction(rec, output_folder):
    rec.write_text(output_folder)
    print(f"Reconstruction exported to {output_folder}")
    with open(output_folder + "/optimized.txt", "w") as f:
        for im in rec.images.values():
            print(f"{im.projection_center()[0]},{im.projection_center()[1]},{im.projection_center()[2]}",file=f)
    with open(output_folder + "/cloud_optimized.txt", "w") as f:
        for p in rec.points3D:
            print(f"{rec.points3D[p].xyz[0]},{rec.points3D[p].xyz[1]},{rec.points3D[p].xyz[2]}",file=f)
    # with open(output_folder + "/initial.txt", "w") as f:
    #     for im in rec_gt.images.values():
    #         print(f"{im.projection_center()[0]},{im.projection_center()[1]},{im.projection_center()[2]}",file=f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)
    args = parser.parse_args()

    # Load configuration from the provided file
    config = configparser.ConfigParser()
    config.read(args.config)

    options = Options(config)
    #options.validate()

    reconstruction = read_reconstruction(options.input_folder)
        
    if options.priors_file:
        priors = read_priors(reconstruction, options)
        if options.transform_reconstruction:
            T = transform_reconstruction(reconstruction, options)
            print("Affine transformation on priors positions:", T)
    problem = define_problem(reconstruction, options)
    solve(problem, options)
    
    write_reconstruction(reconstruction, options.output_folder)
        
if __name__ == "__main__":
    main()