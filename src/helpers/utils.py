import helpers.global_constants as global_constants
import os
import json
import logging
from datetime import datetime


class WalletsListings():
    def __init__(self):
        wallets_availiable = os.listdir(global_constants.GLOBAL_CONSTANTS.path_to_wallets)
        self.wallets_availiable = list(map(lambda file_name: file_name.split(".")[0], wallets_availiable))


def log_and_print(msg : str):
    logging.info(f"{datetime.now().__str__()} {msg}")
    print(msg)


def set_config(new_config, file_path):
    chaning_settings = None
    with open(file_path, "rb") as f:
        chaning_settings = json.load(f)
    
    for key in new_config:
        chaning_settings[key] = new_config[key]
    
    if not f.closed:
        f.close()
        print("Closing file to write to it")

    with open(file_path, "w") as f:
        json.dump(chaning_settings, f)
        f.flush()

def set_wallet_config(new_config, wallet_name):
    wallet_file = os.path.join(global_constants.GLOBAL_CONSTANTS.path_to_wallets, wallet_name + ".json")
    set_config(new_config, wallet_file)


def set_backend_config(new_config):
    set_config(new_config, global_constants.GLOBAL_CONSTANTS.path_to_backend_wabisabi_config)
