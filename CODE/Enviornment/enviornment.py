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
    
    
    
    def step(self, x_cord_pred:float, y_cord_pred:float):
        reward = self.reward(x_cord_pred, y_cord_pred)
        #If object detected
        if self.current_overlap_body:
            self.remove_object(self.current_overlap_body)
        #DO A FORWARD
        self.current_overlap_body = None
        return -1
    

    def update(self):
        self.model, self.data = Enviornment_Randomizer.reset()




    """Transformation methods"""

    def world_to_pixel(self, x_cord:float, y_cord:float) -> tuple[float, float]:
       #we need the z cord since different z_cord's give differnet pixels
       z_cord = self.find_z_at_xy(x=x_cord,y=y_cord)

       #handels if the object is in the deactivated z
       if z_cord == -1:
           return -1.0, -1.0


       # Z cord is manually set, 1.0 is needed for the homogenous coordinate
       world_homogenous = np.array([x_cord,y_cord, z_cord, 1.0])
       #getting camera matrix
       self.calculate_camera_matrix()

       #multiplying camera_m (3x4) by world_homog (4x1)
       pixel_homogenous = self.camera_matrix @ world_homogenous

       #we now have [u × depth, v × depth, depth], so divide by depth
       depth = pixel_homogenous[2]
       if depth <= 0:
            return -1, -1

       pixel_x = pixel_homogenous[0] * 1.0/depth
       pixel_y = pixel_homogenous[1] * 1.0/depth

       return float(pixel_x), float(pixel_y)


    #looks up to find the nearest surface given active z
    def find_z_at_xy(self, x:float, y:float, zstart:float = None) -> float:
        if zstart is None:
            zstart = self.Randomization_Constants.in_scene_z_coordinate
        #making a vector going straight up
        direction_vector = np.array([0.0,0.0,1.0])
        starting_point = np.array([x,y,zstart])

        #saftey
        mujoco.mj_forward(self.model,self.data)
        
        distance = mujoco.mj_ray(self.model, self.data,starting_point, direction_vector,
            None,   # include all geom groups
            True,   # include static geoms
            -1,     # exclude no body
            None,   # do not request geom ID
        )
        if distance == -1:
            return -1
        return zstart + distance


    def remove_object(self,body_name:str):
        self.model.body(body_name).pos = np.array([self.model.body(body_name).pos[0], self.model.body(body_name).pos[1], self.Randomization_Constants.out_of_scene_z_coordinate])
        self.mujoco.mj_forward(self.model, self.data)

    
    """CAMERA METHODS"""

    #calculating the intrisic matrix:
    def calculate_intrisic_matrix(self):
        "USING INTRINSIC MATRIX FORMULA"
        camera_id = self.model.camera(self.Randomization_Constants.camera_name).id

        width, height = self.model.cam_resolution[camera_id]
        fovy_radians = np.deg2rad(float(self.model.cam_fovy[camera_id]))

        f_y = height / (2.0 * np.tan(fovy_radians / 2.0))
        f_x = f_y

        self.intrinsic_matrix = np.array([
            [f_x, 0.0, width / 2.0],
            [0.0, f_y, height / 2.0],
            [0.0, 0.0, 1.0],
        ])

        return self.intrinsic_matrix

    def calculate_extrensic_matrix(self):
        camera_id = self.model.camera(self.Randomization_Constants.camera_name).id
        #for saftey to prevent 0 0 0 
        mujoco.mj_forward(self.model, self.data)

        camera_to_world_rotation = (self.data.cam_xmat[camera_id].reshape(3, 3).copy())
        #for a rotation matrix its inverse is = to its transpose
        world_to_camera_rotation = camera_to_world_rotation.T

        #this is close to the rotation matrix we need HOWEVER Mujoco uses the camera convention where we need to invert z and y axis
        
        #transformation needed to perform what just was mentioned
        axis_conversion = np.diag([1.0, -1.0, -1.0])

        #converting with MM for mujoco precedent
        world_to_camera_rotation = axis_conversion @ world_to_camera_rotation

        #adding offsets 
        camera_position_world = self.data.cam_xpos[camera_id].copy()

        #rotate the camera position to be in this rotated frame
        camera_position_in_rotated_axes = world_to_camera_rotation @ camera_position_world

        #translation is negative since when moving a coordinate system point moves other way
        translation = -1* camera_position_in_rotated_axes

        
        #Takes a 3x3 matrix and concats a 3x1   on the right side to make a 4x3
        self.extrenisic_matrix = np.hstack((world_to_camera_rotation,translation.reshape(3, 1)))
        return self.extrenisic_matrix

        """LOGIC:
        
        I have now rotated it to the proper orentation, but know I need to translate the origin of this coordiante system to the camera coordinate system. 
        Since the axises of the world coordiante system are now rotated how do i find the amount i need to move along each axis to get to the camera point? 
        Is it that i need to rotate the camera point the opposite direction (since rotating a point not a plane) and then use its coordinates as teh translation
        """

    def calculate_camera_matrix(self):
        #Reset both and recalculate just in case
        self.extrenisic_matrix = None
        self.intrinsic_matrix = None
        self.calculate_intrisic_matrix()
        self.calculate_extrensic_matrix()
        #intrisic is 3x3, extrensic is 3x4 so do matrix mult with @
        self.camera_matrix = self.intrinsic_matrix @ self.extrenisic_matrix
        return self.camera_matrix





    """CREATING HEATMAP"""


    """NON HEATMAP TRAINING METHODS:"""

    #method 1: Project camera point to real point and see if it passes conditons and see how close it is: T
        #not really needed because we are doing heatmap
    #returns body name if True

    """ 
    def object_at_cord(self, x_cord: float, y_cord: float, update_current_overlap:bool = False) -> str:
        #reset 
        if update_current_overlap:
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
                    if update_current_overlap:
                        self.current_overlap_body = food_body
                    self.checking_geom_to_pos(x_cord=-1, y_cord=-1,z_cord=-1)
                    return food_body
        
        self.checking_geom_to_pos(x_cord=-1, y_cord=-1,z_cord=-1)
        return None

    #returns precent passed for reward shaping
    def radius_condition(self,x_cord: float, y_cord: float) -> float:
        half_length = self.suction_diameter_in_meters/2
        #points on the circle with radius half_length
        numberOfPoints = 8
        checking_offsets = [
            [half_length, 0, 0],
            [half_length / np.sqrt(2), half_length / np.sqrt(2), 0],
            [0, half_length, 0],
            [-half_length / np.sqrt(2), half_length / np.sqrt(2), 0],
            [-half_length, 0, 0],
            [-half_length / np.sqrt(2), -half_length / np.sqrt(2), 0],
            [0, -half_length, 0],
            [half_length / np.sqrt(2), -half_length / np.sqrt(2), 0],
        ]
        checking_offsets = np.array(checking_offsets)

        numberFailed = 0.0
        for offset in checking_offsets:
            returned_body = self.object_at_cord(x_cord=x_cord, y_cord=y_cord,update_current_overlap = False)
            if returned_body != self.current_overlap_body:
                numberFailed+=1

        return numberFailed*1.0/numberOfPoints


    def distance_to_optimal_position(self,x_cord: float, y_cord:float) -> float:
        if self.current_overlap_body is None:
            raise ValueError("No overlapping food body is currently selected")
        body_name = self.current_overlap_body
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        optimal_point = self.data.subtree_com[body_id, :2]

        #calculate distance between the points
        distance = np.sqrt((x_cord - optimal_point[0])**2 + (y_cord - optimal_point[1])**2)
        return float(distance)

    def checking_geom_to_pos(self, x_cord:float, y_cord:float, z_cord:float):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.Randomization_Constants.checking_geom_name)
            new_geom_x_y_z = np.array([x_cord,y_cord,z_cord])
            self.model.geom_pos[geom_id] = new_geom_x_y_z
            mujoco.mj_forward(self.model, self.data)
    def reward(x_cord_pred: float, y_cord_pred: float) -> float:
            
            return -1
    """