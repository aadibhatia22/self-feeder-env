import mujoco
import numpy as np


class Enviornment_Randomizer:

    def __init__(self):
        pass

    def randomize_color(self, model, geom_name):
        model.geom(geom_name).rgba[:3] = np.random.rand(3)  # Random RGB values
        return model

    def randomize_number_of_objects(
        self,
        list_of_food_object_names,
        model,
        min_objects,
        in_secene_z_cord,
        out_of_scene_z_cord,
    ):
        # Keep a local copy so the caller's original list is not changed.
        list_of_food_object_names = np.asarray(
            list_of_food_object_names
        ).copy()
        if list_of_food_object_names.ndim != 1:
            raise ValueError("list_of_food_object_names must be one-dimensional")
        if not 0 <= min_objects <= list_of_food_object_names.size:
            raise ValueError(
                "min_objects must be between 0 and the number of objects"
            )

        number_of_objects_initilized = 0
        rng = np.random.default_rng()
        while True:
            random_number = rng.random()
            if (
                random_number < 0.5
                and number_of_objects_initilized >= min_objects
            ):
                break
            else:
                size = list_of_food_object_names.size
                if size == 0:
                    break

                # already in index position
                if size <= 1:
                    object_index = 0
                else:
                    object_index = rng.integers(0, size)

                name_of_object = str(
                    list_of_food_object_names[object_index]
                )
                model.body(name_of_object).pos[2] = in_secene_z_cord
                list_of_food_object_names = np.delete(
                    list_of_food_object_names, object_index
                )
                number_of_objects_initilized += 1

        for object_name in list_of_food_object_names:
            model.body(str(object_name)).pos[2] = out_of_scene_z_cord

        return model

    def get_body_geom_world_bounds(self, model, data, body_name):
        """Return the world-space center and dimensions of a body's geoms."""
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

        # Find every geom inside the wrapper body or its child bodies.
        for geom_id in self.get_geom_ids_in_body(model, body_name):
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError(
                    f"Body '{body_name}' contains an infinite plane geom"
                )

            # geom_aabb is [local center, local half dimensions].
            local_center = model.geom_aabb[geom_id, :3]
            local_half_dimensions = model.geom_aabb[geom_id, 3:]
            local_corners = (
                local_center
                + corner_signs * local_half_dimensions
            )

            # Transform the local AABB corners into world coordinates.
            geom_world_position = data.geom_xpos[geom_id]
            geom_world_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            transformed_corners = (
                local_corners @ geom_world_rotation.T
                + geom_world_position
            )
            world_corners.append(transformed_corners)

        world_corners = np.vstack(world_corners)
        world_minimum = np.min(world_corners, axis=0)
        world_maximum = np.max(world_corners, axis=0)
        world_center = (world_minimum + world_maximum) / 2
        world_dimensions = world_maximum - world_minimum

        return world_center, world_dimensions

    def randomize_position_of_objects(
        self,
        list_of_food_object_names,
        model,
        plate_radius=0.075,
        plate_body_name="plate",
        clearance=0.005,
        max_attempts=5000,
    ):
        # pass in the body names
        plate_center = model.body(plate_body_name).pos[:2]
        plate_height = model.body(plate_body_name).pos[2]
        data = mujoco.MjData(model)
        placed_food_object_names = []

        for object_name in list_of_food_object_names:
            object_name = str(object_name)
            Passed = False
            attempts = 0

            # Skip food that randomize_number_of_objects moved out of scene.
            object_center, object_dimensions = self.get_body_geom_world_bounds(
                model, data, object_name
            )
            if object_center[2] < plate_height:
                continue

            obj_x_size = object_dimensions[0]
            obj_y_size = object_dimensions[1]

            # maximum allowable offsets to keep the object within the plate
            max_x_offset = plate_radius - obj_x_size / 2
            max_y_offset = plate_radius - obj_y_size / 2
            if max_x_offset < 0 or max_y_offset < 0:
                raise ValueError(
                    f"Body '{object_name}' is too large for the plate"
                )

            original_position = model.body(object_name).pos.copy()
            while not Passed and attempts < max_attempts:
                attempts += 1

                # random offsets within the allowable range
                random_x_offset = np.random.uniform(
                    -1 * max_x_offset, max_x_offset
                )
                random_y_offset = np.random.uniform(
                    -1 * max_y_offset, max_y_offset
                )

                # adjusting to plate coordinates
                object_x_position = plate_center[0] + random_x_offset
                object_y_position = plate_center[1] + random_y_offset
                model.body(object_name).pos[:2] = [
                    object_x_position,
                    object_y_position,
                ]

                # checking for collisions
                if not self.colision_check(
                    model,
                    object_name,
                    placed_food_object_names,
                    clearance,
                ):
                    Passed = True

            if not Passed:
                model.body(object_name).pos[:] = original_position
                raise RuntimeError(
                    f"Could not place '{object_name}' without overlap after "
                    f"{max_attempts} attempts"
                )

            placed_food_object_names.append(object_name)

        return model

    def get_first_geom_in_body(self, model, body_name):
        # first element
        first_geom_id = self.get_geom_ids_in_body(model, body_name)[0]
        return model.geom(first_geom_id).name

    def colision_check(
        self,
        model,
        object_name,
        list_of_food_object_names,
        clearance=0.005,
    ):
        primary_geom_ids = self.get_geom_ids_in_body(model, object_name)

        list_of_geom_ids = []
        for body in list_of_food_object_names:
            if str(body) == object_name:
                continue
            geom_ids = self.get_geom_ids_in_body(model, str(body))
            list_of_geom_ids.extend(geom_ids)

        # convert to numpy array
        list_of_geom_ids = np.array(list_of_geom_ids, dtype=int)

        # get data
        data = mujoco.MjData(model)
        # loading positions for the distance checks
        mujoco.mj_forward(model, data)

        for primary_geom_id in primary_geom_ids:
            for other_geom_id in list_of_geom_ids:
                distance = mujoco.mj_geomDistance(
                    model,
                    data,
                    primary_geom_id,
                    int(other_geom_id),
                    max(1.0, clearance + 1.0),
                    None,
                )
                if distance <= clearance:
                    return True  # Overlap detected

        return False  # it passed all checks

    def get_geom_ids_in_body(self, model, body_name):
        root_body_id = model.body(body_name).id
        geom_ids = []

        for geom_id in range(model.ngeom):
            current_body_id = int(model.geom_bodyid[geom_id])

            while (
                current_body_id != root_body_id
                and current_body_id != 0
            ):
                current_body_id = int(
                    model.body_parentid[current_body_id]
                )

            if current_body_id == root_body_id:
                geom_ids.append(geom_id)

        if not geom_ids:
            raise ValueError(
                f"Body '{body_name}' has no descendant geoms"
            )

        return geom_ids
