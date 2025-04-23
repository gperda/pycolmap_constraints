import json
import pycolmap

class RigCamera:
    def __init__(self, camera_id, cam_from_rig):
        self.camera_id = camera_id
        self.cam_from_rig= cam_from_rig


class Rig:
    def __init__(self, ref_camera_id):
        self.ref_camera_id = ref_camera_id
        self.cameras = {}

    def add_camera(self, camera):
        self.cameras[camera.camera_id] = camera


def read_rig_configuration(file_path):
    """
    Reads and parses a rig configuration JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of Rig objects.

    Raises:
        ValueError: If the JSON structure is invalid.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Validate the structure of the JSON
        if not isinstance(data, list):
            raise ValueError("The JSON file must contain a list of rig configurations.")

        rigs = []
        for rig_data in data:
            if "ref_camera_id" not in rig_data or "cameras" not in rig_data:
                raise ValueError("Each rig configuration must contain 'ref_camera_id' and 'cameras'.")
            if not isinstance(rig_data["cameras"], list):
                raise ValueError("'cameras' must be a list of camera configurations.")

            # Create a Rig object
            rig = Rig(ref_camera_id=rig_data["ref_camera_id"])

            for camera_data in rig_data["cameras"]:
                if "camera_id" not in camera_data:
                    raise ValueError("Each camera must contain 'camera_id'.")
                if "cam_from_rig_rotation" not in camera_data or "cam_from_rig_translation" not in camera_data:
                    raise ValueError("Each camera must contain 'cam_from_rig_rotation' and 'cam_from_rig_translation'.")

                # Create a Camera object and add it to the Rig
                camera = RigCamera(
                    camera_id=camera_data["camera_id"],
                    cam_from_rig=pycolmap.Rigid3d(pycolmap.Rotation3d(camera_data["cam_from_rig_rotation"]),camera_data["cam_from_rig_translation"])
                )
                rig.add_camera(camera)
            rigs.append(rig)
        return rigs

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")