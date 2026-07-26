import time
import os
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Model.UNet.unet import UNet
from Model.model_constants import Model_Constants
from enviornment_randomizer import Enviornment_Randomizer
from randomization_constants import Randomization_Constants
from enviornment import Enviornment
from pathlib import Path


TMP_DIRECTORY = Path(__file__).parent / "tmp"
CHECKPOINT_DIRECTORY = TMP_DIRECTORY / "UNET_MODEL_OVERFIT"
ENVIRONMENT_DIRECTORY = Path(__file__).resolve().parents[2]
WORLD_XML_FILE = ENVIRONMENT_DIRECTORY / "xml_models" / "world.xml"


"""SOME CODE TO IMPORT MODEL SEED"""
# SEED_FILE = TMP_DIRECTORY / "last_seed.txt"
# def load_last_seed():
#     if not SEED_FILE.exists():
#         raise RuntimeError("SEED_FILE NOT FOUND")
#     return int(SEED_FILE.read_text().strip())

# def save_last_seed(seed):
#     SEED_FILE.write_text(str(seed))
#     # print(f'Saved Seed as: {seed} in {SEED_FILE}')

"""CODE TO IMPORT NUMBER OF TRAINING_ITERATIONS"""
# NUMBER_OF_ITERATIONS_FILE = TMP_DIRECTORY / "training_iterations.txt"

# def load_number_of_iterations():
#     if not NUMBER_OF_ITERATIONS_FILE.exists():
#         raise RuntimeError("NUMBER_OF_ITERATIONS_FILE NOT FOUND")
#     return int(NUMBER_OF_ITERATIONS_FILE.read_text().strip())
    
# def save_number_of_iterations(number_of_iterations):
#     NUMBER_OF_ITERATIONS_FILE.write_text(str(number_of_iterations))
#     # print(f'Saved Num of Itr: {number_of_iterations} in {NUMBER_OF_ITERATIONS_FILE}')

"""CODE TO IMPORT LOWEST AVERAGE LOSS OVER 20 ITERATIONS"""
LOWEST_AVG_LOSS_20_FILE = TMP_DIRECTORY / "lowest_avg_loss_overfit_20.txt"


def load_lowest_avg_loss_20():
    if not LOWEST_AVG_LOSS_20_FILE.exists():
        raise RuntimeError("LOWEST_AVG_LOSS_20_FILE NOT FOUND")

    return float(LOWEST_AVG_LOSS_20_FILE.read_text().strip())


def save_lowest_avg_loss_20(lowest_avg_loss_20):
    LOWEST_AVG_LOSS_20_FILE.write_text(str(lowest_avg_loss_20))
    print(
        f"Saved lowest average loss over 20 iterations as: "
        f"{lowest_avg_loss_20} in {LOWEST_AVG_LOSS_20_FILE}"
    )


def get_localization_metrics(prediction, ground_truth):
    """Measure whether the prediction is high at a target and near a target."""
    with torch.no_grad():
        prediction_probability = torch.sigmoid(prediction).squeeze()
        ground_truth = torch.as_tensor(
            ground_truth,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        target_centers = ground_truth == 1

        if not target_centers.any():
            return None, None

        target_center_probability = prediction_probability[target_centers].mean().item()

        prediction_peak_index = prediction_probability.argmax()
        prediction_peak_y = prediction_peak_index // prediction_probability.shape[1]
        prediction_peak_x = prediction_peak_index % prediction_probability.shape[1]
        target_center_coordinates = torch.nonzero(target_centers)
        peak_distances = torch.sqrt(
            (target_center_coordinates[:, 0] - prediction_peak_y).float() ** 2
            + (target_center_coordinates[:, 1] - prediction_peak_x).float() ** 2
        )
        peak_distance = peak_distances.min().item()

        return target_center_probability, peak_distance



def make_env():
    #seed set to none
    seed = np.random.randint(0,32)
    enviornment_randomizer = Enviornment_Randomizer()
    randomization_constants = Randomization_Constants()
    model_constants = Model_Constants()

    env = Enviornment(xml_file=str(WORLD_XML_FILE), Enviornment_Randomizer=enviornment_randomizer,
                      Randomization_Constants=randomization_constants, Model_Constants=model_constants, starting_seed=seed
                      )
    return env



if __name__ =="__main__":
    env = make_env()
    model = UNet(in_channels=3, num_classes=1, checkpoint_dir=str(CHECKPOINT_DIRECTORY), learning_rate=env.Model_Constants.learning_rate)

    goToSavedModel = False

    if goToSavedModel:
        model = model.load_checkpoint()


    total_training_iterations = 20000
    current_iteration = 0
    """TENSORBOARD CONFIG"""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        "logs",
        "overfit_32",
        f"UNet_{run_id}_lr_{env.Model_Constants.learning_rate}_output_xy_{env.Model_Constants.output_x_dim}_{env.Model_Constants.output_y_dim}_MaxReward_{env.Model_Constants.max_reward}_Sigma_{env.Model_Constants.sigma}",
    )
    writer = SummaryWriter(log_dir)

    loss_list = []

    lowest_loss = load_lowest_avg_loss_20()

    try:
        while current_iteration < total_training_iterations:
            #cycle through every overfit scene equally: 0, 1, ..., 31, then repeat
            curr_seed = current_iteration % 32
            current_iteration += 1
            env.update()
            env.new_scene(seed=curr_seed)
            #currently in height x width x rgb = (1080, 1920, 3)
            model_input = env.observation()
            # y first because rows are y axis
            # resize to (1, 3, y_input_dim, x_input_dim) -> (batchsize, # channels, # rows, # columns)
            model_input = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).float()
            model_input = model_input / 255.0
            model_input = F.interpolate(
                model_input,
                size=(
                    env.Model_Constants.input_y_dim,
                    env.Model_Constants.input_x_dim,
                ),
                mode="bilinear",
                align_corners=False,
            )
            model_input = model_input.to(model.device)

            model_output = model.forward(x = model_input)
            ground_truth = env.get_target_heatmap()
            number_of_objects = float(env.get_number_of_active_food_objects())
            target_center_probability, peak_distance = get_localization_metrics(
                prediction=model_output,
                ground_truth=ground_truth,
            )
            #trains and returns loss
            loss = model.train_step(prediction=model_output, ground_truth=ground_truth,N = number_of_objects)

            #adding the loss to the loss_list
            loss_list.append(loss)

            #saves model every 100 and overwrites the previous save
            if current_iteration % 100 == 0:
                print(f"SAVING MODEL AT ITERATION: {current_iteration}")
                model.save_checkpoint()
                average_loss = sum(loss_list)/20.0
                print(f"LAST LOSS AVG: {average_loss}")

            
            

            if len(loss_list) == 20:
                average_loss = sum(loss_list)/20.0
                if average_loss < lowest_loss:
                     save_lowest_avg_loss_20(average_loss)
                     lowest_loss = average_loss
                del loss_list[0]
            #push the score to the tensorboard's logs for us to see
            writer.add_scalar("Loss", loss, global_step=current_iteration)
            if target_center_probability is not None:
                writer.add_scalar(
                    "Localization/Target center probability",
                    target_center_probability,
                    global_step=current_iteration,
                )
                writer.add_scalar(
                    "Localization/Peak distance pixels",
                    peak_distance,
                    global_step=current_iteration,
                )
            writer.flush()
    finally:
        writer.close()
            
