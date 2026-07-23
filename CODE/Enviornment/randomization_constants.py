import numpy as np


class Randomization_Constants:
    """Public configuration values used by Enviornment_Randomizer."""

    def __init__(self):
        # Food wrapper bodies available in world.xml.
        self.all_food_body_names = [
            "full_slice_instance",
            "deformed_slice_instance",
            "rod_instance",
            "block_instance",
            "full_slice_instance_2",
            "deformed_slice_instance_2",
            "rod_instance_2",
            "block_instance_2",
        ]

        # Use all eight wrapper bodies for every food randomization.
        self.size_randomization_body_names = self.all_food_body_names.copy()
        self.rotation_randomization_body_names = self.all_food_body_names.copy()
        self.number_randomization_body_names = self.all_food_body_names.copy()
        self.position_randomization_body_names = self.all_food_body_names.copy()
        self.color_randomization_body_names = self.all_food_body_names.copy()
        self.visibility_check_body_names = self.all_food_body_names.copy()

        self.total_food_slots = len(self.all_food_body_names)
        self.minimum_objects = 1
        self.maximum_objects = len(self.number_randomization_body_names)
        self.in_scene_z_coordinate = 0.18
        self.out_of_scene_z_coordinate = -0.18

        self.primitive_size_ranges = {
            "sphere": [
                [0.003, 0.016],
            ],
            "capsule": [
                [0.003, 0.009],
                [0.008, 0.025],
            ],
            "ellipsoid": [
                [0.014, 0.035],
                [0.014, 0.038],
                [0.003, 0.012],
            ],
            "cylinder": [
                [0.014, 0.034],
                [0.003, 0.012],
            ],
            "box": [
                [0.012, 0.034],
                [0.010, 0.030],
                [0.005, 0.022],
            ],
        }

        self.single_color_geom_name = "1table_top_geom"
        self.multiple_geom_color_body_names = [
            "base_camera_stand_instance",
            "plate",
        ]
        self.plate_color_geom_name = "plate_visual_geom"

        self.plate_body_name = "plate"
        self.checking_geom_name = "checking_geom"
        self.plate_radius = 0.075
        self.object_clearance = 0.005
        self.position_max_attempts = 5000

        self.plate_edge = "1base_plate_geom"
        self.visibility_plate_geom_name = "plate_visual_geom"
        self.minimum_color_distance = 0.30
        self.visibility_max_attempts = 100

        self.camera_name = "emeet_c960_camera"
        self.camera_position_offset_bounds = [
            [-0.10, 0.05],
            [-0.03, 0.03],
            [-0.10, -0.05],
        ]
        self.camera_rotation_degree_limit = 3
        self.camera_rotation_chance = 0.75

        self.minimum_number_of_lights = 1
        self.maximum_number_of_lights = 3
        self.light_x_offset_range = np.array([-2.0, 2.0])
        self.light_y_offset_range = np.array([-2.0, 2.0])
        self.light_z_offset_range = np.array([1.0, 10.0])