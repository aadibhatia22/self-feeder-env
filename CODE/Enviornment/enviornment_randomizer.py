
import mujoco
import numpy as np
class Enviornment_Randomizer:
    

    
    def __init(self):
        pass 

    def randomize_color (self, model, geom_name):
        model.geom(geom_name).rgba[:3] = np.random.rand(3)  # Random RGB values
        return model

    def randomize_number_of_objects(self, list_of_food_object_names, model, min_objects, in_secene_z_cord, out_of_scene_z_cord):
        number_of_objects_initilized = 0
        while True:
            rng = np.random.default_rng()
            random_number = rng.random()
            if random_number<0.5 and number_of_objects_initilized>=min_objects:
                break
            else:
                if(list_of_food_object_names.ndim != 1):
                    break
                
                size = list_of_food_object_names.size
                #already in index position
                object_index = -1
                if(size <= 1):
                    object_index = 0
                else:
                    object_index = rng.integers(0, size)
                name_of_object = list_of_food_object_names[object_index]
                model.body(name_of_object).pos[2] = in_secene_z_cord
                list_of_food_object_names = np.delete(list_of_food_object_names, object_index)
                number_of_objects_initilized += 1
                

        for object_name in list_of_food_object_names:
            object_index = np.argwhere(list_of_food_object_names == object_name)
            model.body(object_name).pos[2] = out_of_scene_z_cord
            list_of_food_object_names = np.delete(list_of_food_object_names, object_index)
        return model 


    def get_body_geom_world_bounds(self, model, data, body_name):
        """Return the world-space center and dimensions of a body's geoms.

        The body can be a top-level wrapper. All geoms attached to that body
        or any of its descendant bodies are included.
        """
        body_id = model.body(body_name).id
        mujoco.mj_forward(model, data)

        corner_signs = np.array([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ], dtype=float)
        world_corners = []

        for geom_id in range(model.ngeom):
            current_body_id = int(model.geom_bodyid[geom_id])

            # Walk up the hierarchy to determine whether this geom belongs to
            # the requested wrapper body's subtree.
            while current_body_id != body_id and current_body_id != 0:
                current_body_id = int(
                    model.body_parentid[current_body_id]
                )

            if current_body_id != body_id:
                continue

            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError(
                    f"Body '{body_name}' contains an infinite plane geom"
                )

            # geom_aabb is [local_center_xyz, local_half_dimensions_xyz].
            local_center = model.geom_aabb[geom_id, :3]
            local_half_dimensions = model.geom_aabb[geom_id, 3:]
            local_corners = (
                local_center
                + corner_signs * local_half_dimensions
            )

            # Transform every local AABB corner into the world frame.
            geom_world_position = data.geom_xpos[geom_id]
            geom_world_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            transformed_corners = (
                local_corners @ geom_world_rotation.T
                + geom_world_position
            )
            world_corners.append(transformed_corners)

        if not world_corners:
            raise ValueError(
                f"Body '{body_name}' has no descendant geoms"
            )

        world_corners = np.vstack(world_corners)
        world_minimum = np.min(world_corners, axis=0)
        world_maximum = np.max(world_corners, axis=0)
        world_center = (world_minimum + world_maximum) / 2
        world_dimensions = world_maximum - world_minimum

        return world_center, world_dimensions

    def randomize_position_of_objects(self, list_of_food_object_names, model, plate_radius =0.075 , plate_body_name = 'plate'):
        #pass in the body names
        plate_center = model.body(plate_body_name).pos[:2]
        data = mujoco.MjData(model)
        for object_name in list_of_food_object_names:
            # while valid_position == False:
            _, object_dimensions = self.get_body_geom_world_bounds(
                model, data, object_name
            )
            obj_x_size = object_dimensions[0]
            obj_y_size = object_dimensions[1]

            # maximum allowable offsets to keep the object within the plate
            max_x_offset = plate_radius - obj_x_size / 2
            max_y_offset = plate_radius - obj_y_size / 2
            # random offsets within the allowable range
            random_x_offset = np.random.uniform(-1 * max_x_offset, max_x_offset)
            random_y_offset = np.random.uniform(-1 * max_y_offset, max_y_offset)

            #adjusting to plate cordinats
            object_x_position = plate_center[0] + random_x_offset
            object_y_position = plate_center[1] + random_y_offset
            model.body(object_name).pos[:2] = [
                object_x_position,
                object_y_position,
            ]
        return model
    
