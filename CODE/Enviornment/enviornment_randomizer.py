from xml.parsers.expat import model

import mujoco
import numpy as np


class Enviornment_Randomizer:

    def __init__(self):
        self.original_camera_positions = {}
        self.original_camera_quaternions = {}

    """ COLOR RANDOMIZATION"""

    def randomize_color_of_single_geom(self, model, geom_name, isBody = False):
        model.geom(geom_name).rgba[:3] = np.random.rand(3)  # Random RGB values
        return model
    def randomize_color_of_multiple_bodies_with_single_geom(self, model, list_of_body_names):
        for body_name in list_of_body_names:
            first_geom_name = self.get_first_geom_in_body(model, body_name)
            model = self.randomize_color_of_single_geom(model, first_geom_name)
        return model
    def randomize_color_of_multiple_geoms(self, model, list_of_geom_names):
        for geom_name in list_of_geom_names:
            model = self.randomize_color_of_single_geom(model, geom_name)
        return model
    def randomize_color_of_multiple_bodies_with_multiple_geom(self, model, list_of_body_names):
        for body_name in list_of_body_names:
           for geom_id in self.get_geom_ids_in_body(model, body_name):
                geom_name = model.geom(geom_id).name
                model = self.randomize_color_of_single_geom(model, geom_name)
        return model
    #Not used
    def randomize_plate_color(self, model, plate_geom_name):
        # Sample one shade ranging from white to light yellow.
        brightness = np.random.uniform(0.85, 1.0)
        green_offset = np.random.uniform(0.0, 0.04)
        blue_offset = np.random.uniform(green_offset, 0.30)
        plate_color = np.array([
            brightness,
            brightness - green_offset,
            brightness - blue_offset,
        ])

        model.geom(plate_geom_name).rgba[:3] = plate_color

        return model
    """ NUMBER OF OBJECTS RANDOMIZATION"""
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

    """ SIZE RANDOMIZATION"""
    def randomize_size(
        self,
        list_of_food_body_names,
        model,
        primitive_size_ranges,
    ):
        """Defines the components for each primative type"""
        primitive_size_components = {
            "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, 1),
            "capsule": (mujoco.mjtGeom.mjGEOM_CAPSULE, 2),
            "ellipsoid": (mujoco.mjtGeom.mjGEOM_ELLIPSOID, 3),
            "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, 2),
            "box": (mujoco.mjtGeom.mjGEOM_BOX, 3),
        }

        validated_ranges = {}
        for primitive_name, bounds in primitive_size_ranges.items():
            number_of_size_components = primitive_size_components[primitive_name][1]
            bounds = np.asarray(bounds, dtype=float)
            expected_shape = (number_of_size_components, 2)
            if bounds.shape != expected_shape:
                raise ValueError(
                    f"'{primitive_name}' ranges must have shape "
                    f"{expected_shape}"
                )
            if np.any(bounds[:, 0] <= 0):
                raise ValueError(
                    f"'{primitive_name}' minimum sizes must be positive"
                )
            if np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError(
                    f"'{primitive_name}' minimum sizes cannot exceed "
                    "their maximum sizes"
                )
            validated_ranges[primitive_name] = bounds

        rng = np.random.default_rng()
        processed_geom_ids = set()

        for body_name in list_of_food_body_names:
            for geom_id in self.get_geom_ids_in_body(model, str(body_name)):
                if geom_id in processed_geom_ids:
                    continue
                processed_geom_ids.add(geom_id)

                geom_type = model.geom_type[geom_id]
                for primitive_name, (
                    primitive_type,
                    number_of_size_components,
                ) in primitive_size_components.items():
                    if (
                        geom_type == primitive_type
                        and primitive_name in validated_ranges
                    ):
                        bounds = validated_ranges[primitive_name]
                        model.geom_size[
                            geom_id, :number_of_size_components
                        ] = rng.uniform(bounds[:, 0], bounds[:, 1])
                        break

        return model
    """ROTATION RANDOMIZATION"""
    def randomize_rotation(self,model, object_list):
        rng = np.random.default_rng()
        #assume bodies are sent in
        for body in object_list:
            #we only need rotation across z_axis
            angle = rng.uniform(-np.pi,np.pi)
            #quaternion rotation
            model.body(body).quat[:]= [
                np.cos(angle/2),
                0.0,
                0.0,
                np.sin(angle/2)
            ]
        return model
        
    """CAMERA ROTATION RANDOMIZATION"""
    def camera_rotation_randomization(self, model, camera_name, degree_limit, chance_of_rotation=0.75):
        if degree_limit < 0:
            raise ValueError("degree_limit must be nonnegative")
        if not 0 <= chance_of_rotation <= 1:
            raise ValueError("chance_of_rotation must be between 0 and 1")

        radian_limit = degree_limit * np.pi/180
        rng = np.random.default_rng()
        random_number = rng.random()
        isRotation = False
        if random_number < chance_of_rotation:
            isRotation = True

        camera = model.camera(camera_name)
        camera_key = (id(model), camera_name)
        if camera_key not in self.original_camera_quaternions:
            self.original_camera_quaternions[
                camera_key
            ] = camera.quat.copy()
        original_quaternion = self.original_camera_quaternions[camera_key]
        camera.quat[:] = original_quaternion

        if isRotation:
            axis_determiner = rng.random()
            rotation_amount = rng.uniform(-radian_limit, radian_limit)
            half_rotation = rotation_amount / 2
            if axis_determiner<0.33:
                #x axis rotation
                rotation_quaternion = np.array([
                    np.cos(half_rotation),
                    np.sin(half_rotation),
                    0.0,
                    0.0,
                ])
            elif axis_determiner<0.66:
                #y axis rotation
                rotation_quaternion = np.array([
                    np.cos(half_rotation),
                    0.0,
                    np.sin(half_rotation),
                    0.0,
                ])
            else:
                #z axis rotation
                rotation_quaternion = np.array([
                    np.cos(half_rotation),
                    0.0,
                    0.0,
                    np.sin(half_rotation),
                ])

            randomized_quaternion = np.empty(4)
            mujoco.mju_mulQuat(
                randomized_quaternion,
                original_quaternion,
                rotation_quaternion,
            )
            camera.quat[:] = randomized_quaternion

        return model



    """ POSITION RANDOMIZATION"""
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
    
    def randomize_camera_position(self, model, camera_name, offset_bounds):
        camera = model.camera(camera_name)
        camera_key = (id(model), camera_name)
        if camera_key not in self.original_camera_positions:
            self.original_camera_positions[camera_key] = camera.pos.copy()
        original_position = self.original_camera_positions[camera_key]
        offset_bounds = np.asarray(offset_bounds, dtype=float)
        if offset_bounds.shape != (3, 2):
            raise ValueError(
                "offset_bounds must contain [lower, upper] bounds for x, y, and z"
            )
        if np.any(offset_bounds[:, 0] > offset_bounds[:, 1]):
            raise ValueError("Each lower bound must be less than or equal to its upper bound")

        random_offsets = np.random.uniform(
            offset_bounds[:, 0], offset_bounds[:, 1]
        )
        new_pos = original_position + random_offsets
        camera.pos[:] = new_pos
        return model

    """LIGHTING"""
    def randomize_lighting(self, model, min_number_of_lights, max_number_of_lights, x_offset_range, y_offset_range, z_offset_range):
        rng = np.random.default_rng()

        # Keep the total scene illumination stable as the number of active
        # lights changes. Diffuse is the main light, ambient is low fill,
        # and specular adds only moderate highlights.
        min_total_diffuse = 0.55
        max_total_diffuse = 1.00
        min_total_ambient = 0.02
        max_total_ambient = 0.12
        min_total_specular = 0.05
        max_total_specular = 0.30
        min_color_tint = 0.90
        max_color_tint = 1.00

        """NEEDS TO BE MANUALLY SET"""
        max_number_of_lights_allowable = model.nlight
        max_number_of_lights = min(max_number_of_lights_allowable, max_number_of_lights)


        #number of lights being created
        number_of_active_lights = rng.integers(min_number_of_lights, max_number_of_lights+1)
        if x_offset_range.size != 2 or y_offset_range.size != 2 or z_offset_range.size != 2:
            return None

        if number_of_active_lights > 0:
            light_weights = rng.dirichlet(
                np.ones(number_of_active_lights)
            )
            total_diffuse = rng.uniform(
                min_total_diffuse, max_total_diffuse
            )
            total_ambient = rng.uniform(
                min_total_ambient, max_total_ambient
            )
            total_specular = rng.uniform(
                min_total_specular, max_total_specular
            )
        else:
            light_weights = np.array([])

        number_of_lights_changed = 0
        for i in range(number_of_active_lights):
            x_pos = rng.uniform(x_offset_range[0], x_offset_range[1])
            y_pos = rng.uniform(y_offset_range[0], y_offset_range[1])
            z_pos = rng.uniform(z_offset_range[0], z_offset_range[1])
            new_camera_pos = [x_pos, y_pos, z_pos]
            light = model.light(i)
            light.active[0] = 1
            light.pos[:] = new_camera_pos
            light.dir[:] = [0.0,0.0,-1.0]

            # A single physical light uses the same slight RGB tint for its
            # diffuse, ambient, and specular contributions.
            color_tint = rng.uniform(
                min_color_tint, max_color_tint, size=3
            )
            color_tint /= np.max(color_tint)
            light.diffuse[:] = (
                total_diffuse * light_weights[i] * color_tint
            )
            light.ambient[:] = (
                total_ambient * light_weights[i] * color_tint
            )
            light.specular[:] = (
                total_specular * light_weights[i] * color_tint
            )
            number_of_lights_changed+=1

        for i in range (number_of_lights_changed, max_number_of_lights_allowable):
            light = model.light(i)
            light.active[0]=0

        return model


    """HELPER METHODS"""
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

            # geom_aabb is [local center, local half dimensions]. Primitive
            # half dimensions are derived from the current randomized size
            # because geom_aabb retains its original compiled value.
            local_center = model.geom_aabb[geom_id, :3]
            local_half_dimensions = self.get_geom_local_half_dimensions(
                model, geom_id
            )
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
    def get_geom_local_half_dimensions(self, model, geom_id):
        """Return current local half dimensions for a geom."""
        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.repeat(geom_size[0], 3)
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return np.array([
                geom_size[0],
                geom_size[0],
                geom_size[0] + geom_size[1],
            ])
        if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return geom_size[:3].copy()
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return np.array([
                geom_size[0],
                geom_size[0],
                geom_size[1],
            ])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return geom_size[:3].copy()

        # Meshes and other non-randomized geom types retain compiled bounds.
        return model.geom_aabb[geom_id, 3:].copy()
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
    def get_first_geom_in_body(self, model, body_name):
        # first element
        first_geom_id = self.get_geom_ids_in_body(model, body_name)[0]
        return model.geom(first_geom_id).name
    def reset(self, model, data):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return model, data
    def visibility_check(
        self,
        model,
        list_of_body_names,
        plate_geom,
        minimum_color_distance=0.30,
        max_attempts=100,
    ):
        if not 0 <= minimum_color_distance <= np.sqrt(3):
            raise ValueError(
                "minimum_color_distance must be between 0 and sqrt(3)"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        plate_color = model.geom(plate_geom).rgba[:3].copy()

        for body_name in list_of_body_names:
           for geom_id in self.get_geom_ids_in_body(model, body_name):
                geom = model.geom(geom_id)
                geom_name = geom.name

                # Do not recolor the plate itself or invisible collision geoms.
                if geom_name == plate_geom or geom.rgba[3] == 0:
                    continue

                attempts = 0
                color_distance = np.linalg.norm(
                    geom.rgba[:3] - plate_color
                )

                while (
                    color_distance < minimum_color_distance
                    and attempts < max_attempts
                ):
                    model = self.randomize_color_of_single_geom(
                        model, geom_name
                    )
                    attempts += 1
                    color_distance = np.linalg.norm(
                        geom.rgba[:3] - plate_color
                    )

                if color_distance < minimum_color_distance:
                    raise RuntimeError(
                        f"Could not find a visible color for '{geom_name}' "
                        f"after {max_attempts} attempts"
                    )

        return model
