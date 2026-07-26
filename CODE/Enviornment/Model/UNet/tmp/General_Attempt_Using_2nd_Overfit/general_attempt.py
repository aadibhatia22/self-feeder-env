import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from Model.UNet.unet import UNet
from Model.UNet.tmp.UNET_MODEL_OVERFIT_2ND_ROUND.UNetV2Constants import (
    UNetV2Constants,
)
from enviornment import Enviornment
from enviornment_randomizer import Enviornment_Randomizer
from randomization_constants import Randomization_Constants


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent
ENVIRONMENT_DIRECTORY = CHECKPOINT_DIRECTORY.parents[3]
WORLD_XML_FILE = ENVIRONMENT_DIRECTORY / "xml_models" / "world.xml"

SEED_FILE = CHECKPOINT_DIRECTORY / "last_seed.txt"
NUMBER_OF_ITERATIONS_FILE = CHECKPOINT_DIRECTORY / "training_iterations.txt"
LOWEST_AVG_LOSS_100_FILE = CHECKPOINT_DIRECTORY / "lowest_avg_loss_100.txt"
BEST_CHECKPOINT_FILE = CHECKPOINT_DIRECTORY / "best_UNet.pt"
BEST_SEED_FILE = CHECKPOINT_DIRECTORY / "best_last_seed.txt"
BEST_ITERATION_FILE = CHECKPOINT_DIRECTORY / "best_training_iteration.txt"
TENSORBOARD_RUN_NAME_FILE = CHECKPOINT_DIRECTORY / "tensorboard_run_name.txt"
TENSORBOARD_ROOT_DIRECTORY = (
    ENVIRONMENT_DIRECTORY / "logs" / "general_attempt_using_2nd_overfit"
)


def load_last_seed():
    if not SEED_FILE.exists():
        SEED_FILE.write_text("255")

    return int(SEED_FILE.read_text().strip())


def save_last_seed(seed):
    SEED_FILE.write_text(str(seed))


def load_number_of_iterations():
    if not NUMBER_OF_ITERATIONS_FILE.exists():
        NUMBER_OF_ITERATIONS_FILE.write_text("0")

    return int(NUMBER_OF_ITERATIONS_FILE.read_text().strip())


def save_number_of_iterations(number_of_iterations):
    NUMBER_OF_ITERATIONS_FILE.write_text(str(number_of_iterations))


def load_lowest_avg_loss_100():
    if not LOWEST_AVG_LOSS_100_FILE.exists():
        LOWEST_AVG_LOSS_100_FILE.write_text(str(float("inf")))

    return float(LOWEST_AVG_LOSS_100_FILE.read_text().strip())


def save_best_model(model, average_loss_100, seed, number_of_iterations):
    torch.save(model.state_dict(), BEST_CHECKPOINT_FILE)
    LOWEST_AVG_LOSS_100_FILE.write_text(str(average_loss_100))
    BEST_SEED_FILE.write_text(str(seed))
    BEST_ITERATION_FILE.write_text(str(number_of_iterations))
    print(
        f"Saved best model with average loss over 100 iterations: "
        f"{average_loss_100}"
    )


def save_training_state(model, seed, number_of_iterations):
    # Save weights first. The counters should never advance past the checkpoint.
    model.save_checkpoint()
    save_last_seed(seed)
    save_number_of_iterations(number_of_iterations)


def load_or_create_tensorboard_log_directory(model_constants):
    if TENSORBOARD_RUN_NAME_FILE.exists():
        run_name = TENSORBOARD_RUN_NAME_FILE.read_text().strip()
        if run_name:
            return TENSORBOARD_ROOT_DIRECTORY / run_name

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"UNet_{run_id}_lr_{model_constants.learning_rate}_"
        f"output_xy_{model_constants.output_x_dim}_{model_constants.output_y_dim}_"
        f"MaxReward_{model_constants.max_reward}_Sigma_{model_constants.sigma}"
    )
    TENSORBOARD_RUN_NAME_FILE.write_text(run_name)
    return TENSORBOARD_ROOT_DIRECTORY / run_name


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
    last_seed = load_last_seed()
    enviornment_randomizer = Enviornment_Randomizer()
    randomization_constants = Randomization_Constants()
    model_constants = UNetV2Constants()

    return Enviornment(
        xml_file=str(WORLD_XML_FILE),
        Enviornment_Randomizer=enviornment_randomizer,
        Randomization_Constants=randomization_constants,
        Model_Constants=model_constants,
        starting_seed=last_seed + 1,
    )


if __name__ == "__main__":
    env = make_env()
    model = UNet(
        in_channels=3,
        num_classes=1,
        checkpoint_dir=str(CHECKPOINT_DIRECTORY),
        learning_rate=env.Model_Constants.learning_rate,
    )
    model.load_checkpoint()

    current_iteration = load_number_of_iterations()
    last_completed_seed = load_last_seed()

    log_dir = load_or_create_tensorboard_log_directory(env.Model_Constants)
    purge_step = current_iteration + 1 if current_iteration > 0 else None
    writer = SummaryWriter(
        log_dir=str(log_dir),
        purge_step=purge_step,
    )

    print(f"Resuming training at iteration: {current_iteration}")
    print(f"Last completed seed: {last_completed_seed}")
    print(f"Next seed: {last_completed_seed + 1}")
    print(f"TensorBoard log directory: {log_dir}")

    loss_list = []
    lowest_average_loss_100 = load_lowest_avg_loss_100()

    # Always keep a separate best-model fallback, even before the first
    # complete 100-loss block has been measured.
    if not BEST_CHECKPOINT_FILE.exists():
        torch.save(model.state_dict(), BEST_CHECKPOINT_FILE)
        BEST_SEED_FILE.write_text(str(last_completed_seed))
        BEST_ITERATION_FILE.write_text(str(current_iteration))

    try:
        # Continue through new sequential seeds until the process is stopped.
        while True:
            current_seed = last_completed_seed + 1

            env.update()
            env.new_scene(seed=current_seed)

            #currently in height x width x rgb = (1080, 1920, 3)
            model_input = env.observation()
            # y first because rows are y axis
            # resize to (1, 3, y_input_dim, x_input_dim) -> (batchsize, # channels, # rows, # columns)
            model_input = (
                torch.from_numpy(model_input)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
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

            model_output = model(model_input)
            ground_truth = env.get_target_heatmap()
            number_of_objects = float(env.get_number_of_active_food_objects())
            target_center_probability, peak_distance = get_localization_metrics(
                prediction=model_output,
                ground_truth=ground_truth,
            )

            #trains and returns loss
            loss = model.train_step(
                prediction=model_output,
                ground_truth=ground_truth,
                N=number_of_objects,
            )

            current_iteration += 1
            last_completed_seed = current_seed
            loss_list.append(loss)

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

            if len(loss_list) == 100:
                average_loss_100 = sum(loss_list) / 100.0
                writer.add_scalar(
                    "Loss/Average over 100",
                    average_loss_100,
                    global_step=current_iteration,
                )
                if average_loss_100 < lowest_average_loss_100:
                    save_best_model(
                        model=model,
                        average_loss_100=average_loss_100,
                        seed=last_completed_seed,
                        number_of_iterations=current_iteration,
                    )
                    lowest_average_loss_100 = average_loss_100
                loss_list.clear()

            # Save one synchronized resume point every 100 successful scenes.
            if current_iteration % 100 == 0:
                print(f"SAVING MODEL AT ITERATION: {current_iteration}")
                print(f"LAST COMPLETED SEED: {last_completed_seed}")
                save_training_state(
                    model=model,
                    seed=last_completed_seed,
                    number_of_iterations=current_iteration,
                )
                writer.flush()

    except KeyboardInterrupt:
        print("\nStopping training and saving the latest completed scene...")
        save_training_state(
            model=model,
            seed=last_completed_seed,
            number_of_iterations=current_iteration,
        )
        writer.flush()
        print(f"Saved at iteration: {current_iteration}")
        print(f"Saved through seed: {last_completed_seed}")
    finally:
        writer.close()
