import mujoco
import numpy as np


class Enviornment_Randomizer:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def randomize_color(self, model, geom_name):
        model.geom(geom_name).rgba[:3] = self.rng.random(3)
        return model

    def randomize_number_of_objects(
        self,
        list_of_food_object_names,
        model,
        min_objects,
        in_secene_z_cord,
        out_of_scene_z_cord,
    ):
        body_names = np.asarray(list_of_food_object_names, dtype=str)

        if body_names.ndim != 1:
            raise ValueError("list_of_food_object_names must be one-dimensional")
        if not 0 <= min_objects <= body_names.size:
            raise ValueError(
                f"min_objects must be between 0 and {body_names.size}"
            )

        number_to_show = min_objects
        while (
            number_to_show < body_names.size
            and self.rng.random() >= 0.5
        ):
            number_to_show += 1

        selected_names = set(
            self.rng.choice(
                body_names,
                size=number_to_show,
                replace=False,
            ).tolist()
        )

        for body_name in body_names:
            body_name = str(body_name)
            if body_name in selected_names:
                model.body(body_name).pos[2] = in_secene_z_cord
            else:
                model.body(body_name).pos[2] = out_of_scene_z_cord

        return model

    def get_geom_ids_in_body(self, model, body_name):
        """Return IDs for all geoms in a wrapper body's subtree."""
        root_body_id = model.body(body_name).id
        geom_ids = []

        for geom_id in range(model.ngeom):
            current_body_id = int(model.geom_bodyid[geom_id])

            while current_body_id != root_body_id and current_body_id != 0:
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

    def get_body_geom_world_bounds(self, model, data, body_name):
        """Return world-space center and dimensions of all descendant geoms."""
        mujoco.mj_forward(model, data)

        corner_signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        world_corners = []

        for geom_id in self.get_geom_ids_in_body(model, body_name):
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError(
                    f"Body '{body_name}' contains an infinite plane geom"
                )

            local_center = model.geom_aabb[geom_id, :3]
            local_half_dimensions = model.geom_aabb[geom_id, 3:]
            local_corners = (
                local_center + corner_signs * local_half_dimensions
            )

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

    def collision_check(
        self,
        model,
        data,
        object_name,
        list_of_food_object_names,
        clearance=0.005,
    ):
        """Check one wrapper body against the other food-body subtrees."""
        primary_geom_ids = self.get_geom_ids_in_body(
            model, object_name
        )
        mujoco.mj_forward(model, data)

        for other_body_name in list_of_food_object_names:
            other_body_name = str(other_body_name)
            if other_body_name == object_name:
                continue

            other_geom_ids = self.get_geom_ids_in_body(
                model, other_body_name
            )
            for primary_geom_id in primary_geom_ids:
                for other_geom_id in other_geom_ids:
                    distance = mujoco.mj_geomDistance(
                        model,
                        data,
                        primary_geom_id,
                        other_geom_id,
                        max(1.0, clearance + 1.0),
                        None,
                    )
                    if distance <= clearance:
                        return True

        return False

    # Backward-compatible alias for the earlier misspelled method name.
    def colision_check(
        self,
        model,
        object_name,
        list_of_food_object_names,
        clearance=0.005,
    ):
        data = mujoco.MjData(model)
        return self.collision_check(
            model,
            data,
            object_name,
            list_of_food_object_names,
            clearance,
        )

    def randomize_position_of_objects(
        self,
        list_of_food_object_names,
        model,
        plate_radius=0.125,
        plate_body_name="plate",
        clearance=0.005,
        max_attempts=1000,
    ):
        """Place wrapper bodies on the plate without food-food overlap."""
        body_names = [str(name) for name in list_of_food_object_names]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        plate_center = data.body(plate_body_name).xpos[:2].copy()
        plate_height = float(data.body(plate_body_name).xpos[2])
        visible_body_names = []

        for body_name in body_names:
            world_center, _ = self.get_body_geom_world_bounds(
                model, data, body_name
            )
            if world_center[2] >= plate_height:
                visible_body_names.append(body_name)

        placed_body_names = []
        for object_name in visible_body_names:
            _, object_dimensions = self.get_body_geom_world_bounds(
                model, data, object_name
            )

            object_half_dimensions = object_dimensions[:2] / 2
            if np.linalg.norm(object_half_dimensions) > plate_radius:
                raise ValueError(
                    f"Body '{object_name}' is too large for a plate with "
                    f"radius {plate_radius} m"
                )

            original_position = model.body(object_name).pos.copy()
            placed = False

            for _ in range(max_attempts):
                candidate_xy = plate_center + self.rng.uniform(
                    low=-plate_radius + object_half_dimensions,
                    high=plate_radius - object_half_dimensions,
                )

                footprint_corners = candidate_xy + np.array(
                    [
                        [-object_half_dimensions[0], -object_half_dimensions[1]],
                        [-object_half_dimensions[0], object_half_dimensions[1]],
                        [object_half_dimensions[0], -object_half_dimensions[1]],
                        [object_half_dimensions[0], object_half_dimensions[1]],
                    ]
                )
                if np.any(
                    np.linalg.norm(
                        footprint_corners - plate_center,
                        axis=1,
                    ) > plate_radius
                ):
                    continue

                model.body(object_name).pos[:2] = candidate_xy

                if not self.collision_check(
                    model,
                    data,
                    object_name,
                    placed_body_names,
                    clearance,
                ):
                    placed = True
                    break

            if not placed:
                model.body(object_name).pos[:] = original_position
                mujoco.mj_forward(model, data)
                raise RuntimeError(
                    f"Could not place '{object_name}' without overlap after "
                    f"{max_attempts} attempts. Increase plate_radius, reduce "
                    "clearance, or place fewer visible objects."
                )

            placed_body_names.append(object_name)

        return model
