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
