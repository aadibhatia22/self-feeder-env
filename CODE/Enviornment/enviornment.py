import mujoco
import numpy as np
from enviornment_randomizer import Enviornment_Randomizer
from randomization_constants import Randomization_Constants
class Enviornment:

    # def __init__(self, food_object_list, xml_file, enviornment_randomizer, camera_name,
    #              observation_width, observation_height):
    #     self.food_object_list = food_object_list
    #     self.xml_file = xml_file
    #     self.randomizer = enviornment_randomizer
    #     self.camera_name = camera_name
    #     self.observation_width = observation_width
    #     self.observation_height = observation_height

    def __init__(self, xml_file: str , Enviornment_Randomizer: Enviornment_Randomizer, Randomization_Constants: Randomization_Constants, 
                 checking_height:float, suction_diameter_in_meters:float = 0.01): 
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.Randomization_Constants = Randomization_Constants
        self.Enviornment_Randomizer = Enviornment_Randomizer
        self.checking_height = checking_height
        self.suction_diameter_in_meters = suction_diameter_in_meters

    #RETURNS new data and model after applying randomizaion
    def new_scene(self):
        # Do all the randomizations
        constants = self.Randomization_Constants
        randomizer = self.Enviornment_Randomizer

        self.model = randomizer.randomize_color_of_single_geom(
            model=self.model,
            geom_name=constants.single_color_geom_name,
        )

        self.model = randomizer.randomize_size(
            list_of_food_body_names=constants.size_randomization_body_names,
            model=self.model,
            primitive_size_ranges=constants.primitive_size_ranges,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_rotation(
            model=self.model,
            object_list=constants.rotation_randomization_body_names,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_number_of_objects(
            list_of_food_object_names=constants.number_randomization_body_names,
            model=self.model,
            min_objects=constants.minimum_objects,
            in_secene_z_cord=constants.in_scene_z_coordinate,
            out_of_scene_z_cord=constants.out_of_scene_z_coordinate,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_position_of_objects(
            list_of_food_object_names=constants.position_randomization_body_names,
            model=self.model,
            plate_radius=constants.plate_radius,
            plate_body_name=constants.plate_body_name,
            clearance=constants.object_clearance,
            max_attempts=constants.position_max_attempts,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_color_of_multiple_bodies_with_single_geom(
            model=self.model,
            list_of_body_names=constants.color_randomization_body_names,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_color_of_multiple_bodies_with_multiple_geom(
            model=self.model,
            list_of_body_names=constants.multiple_geom_color_body_names,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.visibility_check(
            model=self.model,
            list_of_body_names=constants.visibility_check_body_names,
            plate_geom=constants.visibility_plate_geom_name,
            minimum_color_distance=constants.minimum_color_distance,
            max_attempts=constants.visibility_max_attempts,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_camera_position(
            model=self.model,
            camera_name=constants.camera_name,
            offset_bounds=constants.camera_position_offset_bounds,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.randomize_lighting(
            model=self.model,
            min_number_of_lights=constants.minimum_number_of_lights,
            max_number_of_lights=constants.maximum_number_of_lights,
            x_offset_range=constants.light_x_offset_range,
            y_offset_range=constants.light_y_offset_range,
            z_offset_range=constants.light_z_offset_range,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        self.model = randomizer.camera_rotation_randomization(
            model=self.model,
            camera_name=constants.camera_name,
            degree_limit=constants.camera_rotation_degree_limit,
            chance_of_rotation=constants.camera_rotation_chance,
        )
        self.model, self.data = randomizer.reset(
            model=self.model,
            data=self.data,
        )

        return self.model, self.data


    def observation():
        return -1
    
    def reward(x_cord_pred: float, y_cord_pred: float) -> float:
        
        return -1
    
    def step(self):


        self.current_overlap_body = None
        return -1
    

    def update(self):
        self.model, self.data = Enviornment_Randomizer.reset()
    
    def object_at_cord(self, x_cord: float, y_cord: float) -> bool:
        #reset 
        self.current_overlap_body = None
        #update position of checking geom
        self.checking_geom_to_pos(x_cord=x_cord, y_cord=y_cord,z_cord=self.Randomization_Constants.in_scene_z_coordinate)
        checking_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.Randomization_Constants.checking_geom_name)
        distmax = 1.0 #largest check
        for food_body in self.Randomization_Constants.all_food_body_names:
            dummy_fromto = np.zeros(6)

            for food_geom in self.Enviornment_Randomizer.get_geom_ids_in_body(self.model, food_body):
                # Call the function cleanly
                distance = mujoco.mj_geomDistance(self.model, self.data, checking_geom_id, food_geom, distmax, dummy_fromto)
                if distance<=0:
                    self.current_overlap_body = food_body
                    self.checking_geom_to_pos(x_cord=-1, y_cord=-1,z_cord=-1)
                    return True
        
        self.checking_geom_to_pos(x_cord=-1, y_cord=-1,z_cord=-1)
        return False

    
    def radius_condition(x_cord: float, y_cord: float) -> bool:
        return -1
    
    def scale_for_optimal_position(x_cord: float, y_cord:float) -> float:
        return -1

    def pixel_to_cordinates(self, x_pixel:float, y_pixel:float) -> tuple[float, float]:
        if not np.isfinite(x_pixel) or not np.isfinite(y_pixel):
            raise ValueError("Pixel coordinates must be finite")

        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.Randomization_Constants.camera_name,
        )
        if camera_id == -1:
            raise ValueError(
                f"Camera '{self.Randomization_Constants.camera_name}' "
                "does not exist in the model"
            )

        image_width, image_height = self.model.cam_resolution[camera_id]
        if not 0 <= x_pixel < image_width or not 0 <= y_pixel < image_height:
            raise ValueError(
                f"Pixel must be inside the {image_width}x{image_height} image"
            )

        mujoco.mj_forward(self.model, self.data)

        vertical_fov = np.deg2rad(self.model.cam_fovy[camera_id])
        half_image_height = np.tan(vertical_fov / 2.0)
        aspect_ratio = image_width / image_height
        half_image_width = aspect_ratio * half_image_height

        normalized_x = (2.0 * x_pixel / image_width) - 1.0
        normalized_y = 1.0 - (2.0 * y_pixel / image_height)

        camera_ray = np.array(
            [
                normalized_x * half_image_width,
                normalized_y * half_image_height,
                -1.0,
            ],
            dtype=np.float64,
        )

        camera_rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        world_ray = camera_rotation @ camera_ray
        camera_position = self.data.cam_xpos[camera_id]

        if np.isclose(world_ray[2], 0.0):
            raise ValueError("The camera ray is parallel to the active plane")

        active_z = self.Randomization_Constants.in_scene_z_coordinate
        ray_scale = (
            active_z - camera_position[2]
        ) / world_ray[2]

        if ray_scale < 0:
            raise ValueError("The active plane is behind the camera")

        world_point = camera_position + ray_scale * world_ray

        return float(world_point[0]), float(world_point[1])

   

    def remove_object():

        return -1
    

    def checking_geom_to_pos(self, x_cord:float, y_cord:float, z_cord:float):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.Randomization_Constants.checking_geom_name)
        new_geom_x_y_z = np.array([x_cord,y_cord,z_cord])
        self.model.geom_pos[geom_id] = new_geom_x_y_z
        mujoco.mj_forward(self.model, self.data)
