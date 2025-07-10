import json
import os

class GlobalConstants():
    backend_endpoint = "http://localhost:37127/"
    client_endpoint = "http://127.0.0.1:37128/"

    bitcoin_testnet_rpc_url = "http://127.0.0.1:18332/"
    bitcoin_regtest_rpc_url = "http://127.0.0.1:18443/"

    network = "RegTest" # one of RegRest, TestNet

    backend_folder_path = "..\\WalletWasabi-v2.0.4\\WalletWasabi.Backend"
    client_folder_path = "..\\WalletWasabi-v2.0.4\\WalletWasabi.Daemon"
    distributor_wallet_name = "DistributorWallet"
    version2 = True

    wasabi_wallet_data = ""

    # time needed in registration phase of round to start scenarion in it
    starting_round_time_required = 60

    # in wallet wasabi, changes in configurations are checked every 10 seconds
    config_refresh_time = 10

    coin_tresholds = 5000

    rpc_user = ""
    rpc_pswd = ""

    def __init__(self):
        self.path_to_client_data = os.path.join(self.wasabi_wallet_data, "Client")
        self.path_to_wallets = os.path.join(self.path_to_client_data, "Wallets", self.network)
        self.path_to_backend = os.path.join(self.wasabi_wallet_data, "Backend")
        self.path_to_backend_wabisabi_config = os.path.join(self.path_to_backend, "WabiSabiConfig.json")
        self.path_to_prison = os.path.join(self.path_to_backend, "WabiSabi", "Prison.txt")

        self.config_path = "{}/Client/Config.json".format(self.wasabi_wallet_data)
        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                loaded_config = json.load(f)
                self.rpc_user = loaded_config["JsonRpcUser"]
                self.rpc_pswd = loaded_config["JsonRpcPassword"]
        else:
            print('WARNING: {} not found'.format(self.config_path))


GLOBAL_CONSTANTS = GlobalConstants()

