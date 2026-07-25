import time
import os
import mujoco
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from unet import UNet
from model_constants import Model_Constants
from enviornment_randomizer import Enviornment_Randomizer
from randomization_constants import Randomization_Constants
from enviornment import Enviornment
from pathlib import Path


"""SOME CODE TO IMPORT MODEL SEED"""
SEED_FILE = Path(__file__).with_name("last_seed.txt")
def load_last_seed():
    if not SEED_FILE.exists():
        raise RuntimeError("SEED_FILE NOT FOUND")
    return int(SEED_FILE.read_text().strip())

def save_last_seed(seed):
    SEED_FILE.write_text(str(seed))
    print(f'Saved Seed as: {seed} in {SEED_FILE}')

"""CODE TO IMPORT NUMBER OF TRAINING_ITERATIONS"""
NUMBER_OF_ITERATIONS_FILE = Path(__file__).with_name("training_iterations.txt")

def load_number_of_iterations():
    if not NUMBER_OF_ITERATIONS_FILE.exists():
        raise RuntimeError("NUMBER_OF_ITERATIONS_FILE NOT FOUND")
    return int(NUMBER_OF_ITERATIONS_FILE.read_text().strip())
    
def save_number_of_iterations(number_of_iterations):
    NUMBER_OF_ITERATIONS_FILE.write_text(str(number_of_iterations))
    print(f'Saved Seed as: {NUMBER_OF_ITERATIONS_FILE} in {NUMBER_OF_ITERATIONS_FILE}')


def make_env():
    #seed set to none
    last_seed = load_last_seed()
    Enviornment_Randomizer = Enviornment_Randomizer()
    Randomization_Constants = Randomization_Constants()
    Model_Constants = Model_Constants()

    env = Enviornment(xml_file="../xml_models/world.xml", Enviornment_Randomizer=Enviornment_Randomizer,
                      Randomization_Constants=Randomization_Constants, Model_Constants = Model_Constants,starting_seed=last_seed + 1
                      )
    return env



if __name__ =="__main__":
    env = make_env()
