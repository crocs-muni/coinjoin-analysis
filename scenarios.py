import requests
import time
import os
import json
from typing import List
import logging
from datetime import datetime, timedelta
import rpc_commands
import regtest_control
import sys
import re
from dateutil import tz
import global_constants
import processes_control


def log_and_print(msg : str):
    logging.info(f"{datetime.now().__str__()} {msg}")
    print(msg)


class ConstantsScenarios():
    backend_url = "http://localhost:37127/"
    url = "http://127.0.0.1:37128/"
    bitcoin_testnet_rpc_url = "http://127.0.0.1:18332/"
    bitcoin_regtest_rpc_url = "http://127.0.0.1:18443/"
    coin_tresholds = 2000
    distributor_wallet = "DistributorWallet"
    network = "RegTest"  # one of RegTest, TestNet, MainNet
    version2 = True

    # time needed in registration phase of round to start scenarion in it
    starting_round_time_required = 60

    # in wallet wasabi, changes in configurations are checked every 10 seconds
    config_refresh_time = 10 

    def __init__(self):
        # works only on windows, need to be expanded for other OS to get application data folder
        self.path_to_app_data = os.getenv('APPDATA')

        self.path_to_client_data = os.path.join(self.path_to_app_data, "WalletWasabi", "Client")
        self.path_to_wallets = os.path.join(self.path_to_client_data, "Wallets", self.network)
        self.path_to_backend = os.path.join(self.path_to_app_data, "WalletWasabi", "Backend")
        self.path_to_backend_wabisabi_config = os.path.join(self.path_to_backend, "WabiSabiConfig.json")


CONSTANTS_SCENARIOS = ConstantsScenarios()


class ScenarioManager():
    round_count = 0
    participants_count = 0
    starting_funds = []
    fresh_wallets = True

    starting_baceknd_config = None
    starting_wallets_config = None

    used_wallets = []

    initial_backend_settings_timestamp = None

    def progress_round(self):
        pass


def translate_json_name(name: str):
    if global_constants.GLOBAL_CONSTANTS.version2:
        result = str(name[0]).upper() + name[1:]
    else:
        result = str(name[0]).lower() + name[1:]
    return result


class JsonParamsNaming():
    def __init__(self):
        self.round_id = translate_json_name("roundId")
        self.round_states = translate_json_name("roundStates")
        self.state_id = translate_json_name("stateId")
        self.phase = translate_json_name("phase")
        self.input_registration = translate_json_name("inputRegistration")
        self.input_registration_start = translate_json_name("inputRegistrationStart")
        self.input_registration_remaining = translate_json_name("inputRegistrationRemaining")
        self.is_blame_round = translate_json_name("isBlameRound")
        self.id = translate_json_name("id")

backend_names = JsonParamsNaming()


class SimplePassiveScenarioManager(ScenarioManager):
    def __init__(self, 
                 round_count: int,
                 participants_count: int,
                 backend_config,
                 initial_backend_settings_timestamp: datetime,
                 wallets_config,
                 used_wallets: List[str]):

        self.round_count = round_count
        self.participants_count = participants_count
        self.starting_baceknd_config = backend_config
        self.starting_wallets_config = wallets_config

        self.initial_backend_settings_timestamp = initial_backend_settings_timestamp

        self.used_wallets = used_wallets

    def progress_round(self):
        return


class WalletsListings():
    def __init__(self):
        wallets_availiable = os.listdir(global_constants.GLOBAL_CONSTANTS.path_to_wallets)
        self.wallets_availiable = list(map(lambda file_name: file_name.split(".")[0], wallets_availiable))


class Coin():
    def __init__(self, txid, index, amount):
        self.txid = txid
        self.index = index
        self.amount = amount


class CoinJoinWalletManager():
    def __init__(self, wallets: List[str]) -> None:
        self.wallets: List[str] = wallets

    def start_coinjoin(self, wallet):
        if global_constants.GLOBAL_CONSTANTS.version2:
            rpc_commands.confirmed_load(wallet)
            rpc_commands.start_coinjoin(wallet, verbose=False)
        else:
            rpc_commands.confirmed_select(wallet)
            rpc_commands.start_coinjoin(verbose=False)

    def stop_coinjoin(self, wallet):
        if global_constants.GLOBAL_CONSTANTS.version2:
            rpc_commands.confirmed_load(wallet)
            rpc_commands.stop_coinjoin(wallet)
        else:
            rpc_commands.confirmed_select(wallet)
            rpc_commands.stop_coinjoin(verbose=False)

    def start_all(self):
        for wallet in self.wallets:
            self.start_coinjoin(wallet)

    def stop_all(self):
        for wallet in self.wallets:
            self.stop_coinjoin(wallet)


def parse_time_to_seconds(time_string: str):
    splitted = time_string.split(" ")
    total_seconds = 0
    for part in splitted:
        unit = part[-1]
        value = int(part[:-1])
        if unit == "d":
            total_seconds += value * 86_400
        elif unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value
    
    return total_seconds


def parse_time_from_backend_response(time_string: str):
    
    string_date, rest = time_string.split("T")
    
    splited_date = string_date.split("-")
    
    string_time = rest.split(".")[0]  # getting rid of miliseconds, not needed here
    splited_time = string_time.split(":")

    constructed_datetime = datetime(
        int(splited_date[0]),
        int(splited_date[1]),
        int(splited_date[2]),
        int(splited_time[0]),
        int(splited_time[1]),
        int(splited_time[2]),
        tzinfo= tz.tzutc()
    )
    constructed_datetime = constructed_datetime.astimezone(tz.tzlocal())
    return constructed_datetime


class RoundChecker():

    started_rounds_tracked = 0
    finished_rounds_tracekd = 0

    rounds_finished = []
    rounds_finished_ids = set()
    rounds_active = []
    rounds_active_ids = set()

    def __init__(self, scenario_manager: ScenarioManager, coin_join_wallet_manager: CoinJoinWalletManager):
        self.round_limit = scenario_manager.round_count

        self.scenario_manager = scenario_manager
        self.coin_join_wallet_manager = coin_join_wallet_manager

    def get_current_rounds(self):
        response = requests.get(global_constants.GLOBAL_CONSTANTS.backend_endpoint + "WabiSabi/human-monitor")
        response_json = response.json()
        return response_json[backend_names.round_states]

    def get_status_from_wasabi_api(self, round_id):
        if global_constants.GLOBAL_CONSTANTS.version2:
            content = "{{\"RoundCheckpoints\": [{{\"RoundId\": \"{0}\", \"StateId\": 0}}]}}".format(round_id)
        else:
            content = "{{\"roundCheckpoints\": [{{\"roundId\": \"{0}\", \"stateId\": 0}}]}}".format(round_id)
        response = requests.post(global_constants.GLOBAL_CONSTANTS.backend_endpoint + "WabiSabi/status", 
                                 data = content,
                                 headers= {
                                     "Accept" : "application/json",
                                     "Content-Type" : "application/json-patch+json"
                                 })
        response_json = response.json()
        return response_json[backend_names.round_states]

    def start(self):
        time_to_compare_rounds = self.scenario_manager.initial_backend_settings_timestamp + timedelta(seconds=global_constants.GLOBAL_CONSTANTS.config_refresh_time)


        from_setting_backend = datetime.now(tz=tz.tzlocal()) - self.scenario_manager.initial_backend_settings_timestamp 
        # enforcing that new backend configurations were noticed and loaded
        while from_setting_backend.seconds < global_constants.GLOBAL_CONSTANTS.config_refresh_time:
            log_and_print("Configuration for backend was not loaded yet. Waiting for other {} seconds".format(global_constants.GLOBAL_CONSTANTS.config_refresh_time - from_setting_backend.seconds))
            time.sleep(global_constants.GLOBAL_CONSTANTS.config_refresh_time - from_setting_backend.seconds)
            from_setting_backend = datetime.now(tz=tz.tzlocal()) - self.scenario_manager.initial_backend_settings_timestamp

        # should not be needed as the config is loaded before backend is started
        # # how to ensure, that current round is not round with previous settings?
        # while True:
        #     current_rounds = self.get_current_rounds()
        #     current_rounds = list(filter(lambda x: x[backend_names.phase] == backend_names.input_registration, current_rounds))
        #     # if there are no rounds running, no problem here
        #     if len(current_rounds) == 0:
        #         break

        #     first_round_id = current_rounds[0][backend_names.round_id]

        #     # get status for rounds
        #     round_states = self.get_status_from_wasabi_api(first_round_id)
        #     # log_and_print("At scenario initiation, there are these rounds: ")

        #     all_rounds_ok = True

        #     # check if all rounds started after loading the initial configuration + refresh time
        #     for round in current_rounds:

        #         correct_round_state = list(filter(lambda x: x[backend_names.id] == round[backend_names.round_id], round_states))[0]
        #         round_started = correct_round_state[backend_names.input_registration_start]

        #         # internet says this should work since python 3.7 but it does not
        #         # real_time_start = datetime.fromisoformat(round_started)
        #         real_time_start = parse_time_from_backend_response(round_started)

        #         if real_time_start < time_to_compare_rounds:
        #             log_and_print("At the start of scenario, at least round {} runs with \
        #                           old configuration for additional {}".format(round[backend_names.round_id], 
        #                                                                       round[backend_names.input_registration_remaining]))
        #             waiting_time = parse_time_to_seconds(round[backend_names.input_registration_remaining])
        #             log_and_print(f"Waiting for {waiting_time} seconds")
        #             all_rounds_ok = False
        #             time.sleep(waiting_time)
        #             break
                
        #     if all_rounds_ok:
        #         break
                

        # ensuring, that there is enaught time in at least one round to register inputs
        registerable_round_long_enaught = False
        while True:
            current_rounds = self.get_current_rounds()
            for round in current_rounds:
                if (round[backend_names.phase] == backend_names.input_registration 
                    and not round[backend_names.is_blame_round]):
                    registerable_round_long_enaught = (registerable_round_long_enaught or
                                                        parse_time_to_seconds(round[backend_names.input_registration_remaining]) 
                                                        >= global_constants.GLOBAL_CONSTANTS.starting_round_time_required)
                    
                    if registerable_round_long_enaught:
                        break
            
            if not registerable_round_long_enaught:
                log_and_print("Waiting for round in which inputs can be registered.")
                time.sleep(5)
            else:
                break
        
        self.run()


    def run(self):

        log_and_print(" Started tracking rounds for scenario")
        
        self.coin_join_wallet_manager.start_all()

        while len(self.rounds_finished) < self.round_limit:
            current_rounds = self.get_current_rounds()
            current_rounds_ids = set(map(lambda x: x[backend_names.round_id], current_rounds))
            ended_rounds = []

            for round in self.rounds_active:
                if round[backend_names.round_id] not in current_rounds_ids:
                    log_and_print(" Registered end of round {0}".format(round[backend_names.round_id]))
                    ended_rounds.append(round)
                    self.rounds_finished.append(round)
                    self.rounds_finished_ids.add(round[backend_names.round_id])
                    log_and_print(" This was round number {}".format(self.finished_rounds_tracekd))
                    self.finished_rounds_tracekd += 1

            for round in ended_rounds:
                self.rounds_active.remove(round)
                self.rounds_active_ids.remove(round[backend_names.round_id])
                
                # if round ended, signall it to manager so it can change configurations if needed
                self.scenario_manager.progress_round()

            
            for round in current_rounds:
                if round[backend_names.round_id] not in self.rounds_active_ids:
                    self.rounds_active.append(round)
                    self.rounds_active_ids.add(round[backend_names.round_id])
                    log_and_print(" Registered creation of round {0}".format(round[backend_names.round_id]))
                    log_and_print(" This is round number {}".format(self.started_rounds_tracked))
                    self.started_rounds_tracked += 1
            
            if len(ended_rounds) > 0:
                regtest_control.mine_block_regtest()
                log_and_print(" Mined new block, because at least one round ended")

            time.sleep(2)
        
        self.coin_join_wallet_manager.stop_all()


def parse_unspent_wallet_coins(trehsold = 2000, wallet: str = global_constants.GLOBAL_CONSTANTS.distributor_wallet_name):
    parsed_coins = []
    if global_constants.GLOBAL_CONSTANTS.version2:
        coins = rpc_commands.list_unspent(wallet, verbose=False)["result"]
    else:
        coins = rpc_commands.list_unspent(verbose=False)["result"]
    for coin in filter(lambda x: x['confirmed'] and x['amount'] > trehsold, coins):
        #print(coin["amount"])
        parsed_coins.append(Coin(coin["txid"], coin["index"], coin["amount"]))
    return parsed_coins


def check_wallet_funds(wallet, needed_coins, amount):
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(wallet)
    else:
        rpc_commands.confirmed_select(wallet)
    wallet_coins = parse_unspent_wallet_coins(wallet=wallet)
    suma = sum(map(lambda coin: coin.amount, wallet_coins))
    print(wallet, suma)
    if suma < amount:
        if global_constants.GLOBAL_CONSTANTS.version2:
            address = rpc_commands.get_address(wallet=wallet)
        else:
            address = rpc_commands.get_address()
        needed_coins.append((address, amount - suma))


def distribute_coins(distributor_coins, requested_funds, verbose = True):
    # building payments field of rpc request from requested funds
    payments = ",\n"
    payments = payments.join("{{\"sendto\":\"{0}\", \"amount\":{1}, \"label\":\"redistribution\"}}".format(request[0], request[1]) for request in requested_funds)
    #print(payments)
    payments = "[\n" + payments + "],"

    payments = ""
    for request in requested_funds:
        if len(payments) != 0:
            payments = payments + ","
        payments = payments + "{{\"sendto\":\"{0}\", \"amount\":{1}, \"label\":\"redistribution\"}}".format(request[0], request[1])

    payments = "[" + payments + "],"

    # calculating how many distributor coins will be needed to fill needs
    needed_amount = sum(map(lambda requested: requested[1], requested_funds))
    ordered_distributor_coins = sorted(distributor_coins, key= lambda coin: coin.amount)
    print(ordered_distributor_coins)
    needed_distributor_coins = 0
    acumulated = 0
    while acumulated <= needed_amount + global_constants.GLOBAL_CONSTANTS.coin_tresholds:
        acumulated += ordered_distributor_coins[needed_distributor_coins].amount
        needed_distributor_coins += 1
    
    # creating used coins 
    coins = ",\n"
    coins = coins.join("{{\"transactionid\":\"{0}\", \"index\":{1}}}".format(ordered_distributor_coins[index].txid, 
                                                                         ordered_distributor_coins[index].index) 
                                                                         for index in range(needed_distributor_coins))

    coins = ""
    for index in range(needed_distributor_coins):
        if index != 0:
            coins = coins + ','
        coins = coins + "{{\"transactionid\":\"{0}\", \"index\":{1}}}".format(ordered_distributor_coins[index].txid, 
                                                                         ordered_distributor_coins[index].index)


    coins = "[" + coins + "],"

    send_content_visible ='''
    {"jsonrpc":"2.0","id":"1","method":"send", "params": 
    { "payments":''' + payments + ''' 
    "coins":''' + coins + '''
    "feeTarget":2, "password": "pswd" }}
    '''

    # "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"selectwallet\", \"params\" : [\"" + wallet_name + "\", \"pswd\"]}"
    # previous version for some reason didnt work with processes started as subprocesses
    send_content = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"send\", \"params\": { \"payments\":" + payments + " \"coins\":" + coins + "\"feeTarget\":2, \"password\": \"pswd\" } }"

    if verbose:
        print(send_content)
    
    print("Before sending funds via command.")
    if global_constants.GLOBAL_CONSTANTS.version2:
        response = rpc_commands.send_post(data = send_content, specified_wallet = global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    else:
        response = rpc_commands.send_post(data = send_content)
    print("Redistribution completed.")
    if verbose:
        print(response.json()) 


def prepare_wallets(needed_coins):

    #needed_amount = sum(map(lambda requested: requested[1], needed_coins))
    #check_distributor(needed_amount)

    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    else:
        rpc_commands.confirmed_select(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)

    distributor_coins = parse_unspent_wallet_coins(wallet=global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    distribute_coins(distributor_coins, needed_coins)

    if global_constants.GLOBAL_CONSTANTS.network == "RegTest":
        regtest_control.mine_block_regtest()
    else:
        # TODO, need to wait until next block is mined
        pass


def prepare_wallets_amount(wallets, amount = 20000):
    needed_coins = []
    for wallet in wallets:
        check_wallet_funds(wallet, needed_coins, amount)
    
    if len(needed_coins) > 0:
        prepare_wallets(needed_coins)


def prepare_wallets_values(wallets, values, default_values = [100000, 50000] ):
    needed_coins = []
    for index, wallet in enumerate(wallets):
        
        if not global_constants.GLOBAL_CONSTANTS.version2:
            rpc_commands.confirmed_select(wallet)
        
        if index in values:
            for value in values[index]:
                # new address each time is needed as if the same value was used, the TXOs would be joined
                if global_constants.GLOBAL_CONSTANTS.version2:
                    address = rpc_commands.get_address(wallet=wallet)
                else:
                    address = rpc_commands.get_address()
                needed_coins.append((address, value))
        
        else:
            for value in default_values:
                # new address each time is needed as if the same value was used, the TXOs would be joined
                if global_constants.GLOBAL_CONSTANTS.version2:
                    address = rpc_commands.get_address(wallet=wallet)
                else:
                    address = rpc_commands.get_address()
                needed_coins.append((address, value))

    if len(needed_coins) > 0:
        prepare_wallets(needed_coins)


def load_scenario():
    scenario = {}
    with open("scenario.json", "r") as f:
        scenario = json.load(f)
    return scenario


def set_config(new_config, file_path):
    chaning_settings = None
    with open(file_path, "rb") as f:
        chaning_settings = json.load(f)

    for key in new_config:
        chaning_settings[key] = new_config[key]

    with open(file_path, "w") as f:
        json.dump(chaning_settings, f)


def set_wallet_config(new_config, wallet_name):
    wallet_file = os.path.join(global_constants.GLOBAL_CONSTANTS.path_to_wallets, wallet_name + ".json")
    set_config(new_config, wallet_file)


def set_backend_config(new_config):
    set_config(new_config, global_constants.GLOBAL_CONSTANTS.path_to_backend_wabisabi_config)


def load_from_scenario(value, default, scenario):
    if value not in scenario:
        log_and_print("'{0}' setting not specified in scenario, defaulting to {}".format(value, default))
        return default
    return scenario[value]


def create_n_same_wallets(count, previous_index_number, wallets_config):
    created_wallets = []

    for i in range(count):
        wallet_name = f"SimplePassiveWallet{previous_index_number + i}"
        log_and_print("Creating wallet " + wallet_name)
        rpc_commands.create_wallet(wallet_name)
        if wallets_config is not None:
            if global_constants.GLOBAL_CONSTANTS.version2:
                rpc_commands.confirmed_load(wallet_name)
            else:
                rpc_commands.confirmed_select(wallet_name)
            set_wallet_config(wallets_config, wallet_name)
        created_wallets.append(wallet_name)

    return created_wallets


def parse_compound_funds(wallets_info, max_allowed_index):
    compound_funds = {}
    for info in wallets_info:
        index = info["walletIndex"]
        funds = info["walletFunds"]
        if index > max_allowed_index:
            raise RuntimeError(f"Index {index} is larger than number of participants. Max allowed index: {max_allowed_index}.")
        if index in compound_funds:
            raise RuntimeError(f"Multiple occurences of index {index} while loading starting wallet funds.")
        compound_funds[index] = funds
    
    return compound_funds


def prepare_simple_scenario(scenario):

    # extracting scenario configurations
    fresh_wallets = load_from_scenario("freshWallets", True, scenario)
    starting_funds = load_from_scenario("startingFunds", [1000000], scenario)
    if not fresh_wallets and "startingFunds" in scenario:
        log_and_print("Starting funds were set but option for creating fresh wallets is turned off, ignoring starting funds option.")
    rounds = load_from_scenario("rounds", 3, scenario)
    participants = load_from_scenario("walletsCounts", 4, scenario)

    backend_config = load_from_scenario("backendConfig", None, scenario)
    if backend_config is None:
        log_and_print("Backend configuration was not specified, using current configuration.")

    wallets_config = load_from_scenario("walletsConfig", None, scenario)
    if wallets_config is None:
        log_and_print("Wallets configuration was not specified, using present configrations of wallets.")

    backend_settings_set_time = None
    if backend_config is not None:
        set_backend_config(backend_config)
    backend_settings_set_time = datetime.now(tz= tz.tzlocal())


    # preparing compound funds if present
    compound_funds = {}
    wallets_info = load_from_scenario("walletsInfo", [], scenario)
    if len(wallets_info) > 0:
        compound_funds = parse_compound_funds(wallets_info, participants - 1)

    # finding out existing wallets and extracting current number to use in name
    existing_wallets = list(filter(lambda x: "SimplePassiveWallet" in x, WalletsListings().wallets_availiable))
    # sorting by number, that is after 'SimplePassiveWallet' text
    existing_wallets.sort(key= lambda x: int(x[19:]))
    if len(existing_wallets) > 0:
        last_wallet = existing_wallets[-1]
        numbers = re.findall(r'\d+$', last_wallet)
        if len(numbers) == 0:
            wallet_number = 1
        else:
            wallet_number = int(numbers[-1]) + 1
    else:
        wallet_number = 1

    used_wallets = []


    # creating needed wallets
    if fresh_wallets:
        log_and_print("Creating fresh wallets.")
        used_wallets = create_n_same_wallets(participants, wallet_number, wallets_config)
        log_and_print("Created these fresh wallets: " + ", ".join(used_wallets))

    else:
        number_of_existing = len(existing_wallets)
        if number_of_existing >= participants:
            used_wallets = existing_wallets[:participants]
            log_and_print("Loaded these existing wallets: " + ", ".join(used_wallets))
        else:
            used_wallets = existing_wallets.copy()
            log_and_print("Loaded these existing wallets: " + ", ".join(used_wallets))
            log_and_print(f"Creating {participants - number_of_existing} new wallets.")

            new_wallets_created = create_n_same_wallets(participants - number_of_existing, wallet_number, wallets_config)
            used_wallets.extend(new_wallets_created)


    # loading all wallets used in the scenario
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
        for wallet in used_wallets:
            rpc_commands.confirmed_load(wallet)

    # distributing funds to wallets used in this scenario
    if fresh_wallets:
        print("Started funds redistribution")
        prepare_wallets_values(used_wallets, compound_funds, starting_funds)
        print("Coins redistribution finished.")
    else:
        prepare_wallets_amount(used_wallets, rounds * 10000)

    scenario_manager = SimplePassiveScenarioManager(rounds,
                                             participants,
                                             backend_config,
                                             backend_settings_set_time,
                                             wallets_config,
                                             used_wallets)
    return scenario_manager


if __name__ == "__main__":


    if global_constants.GLOBAL_CONSTANTS.version2 != CONSTANTS_SCENARIOS.version2:
        print("Sync your versions.")
        sys.exit(1)

    logging.basicConfig(filename='automation.log', encoding='utf-8', level=logging.INFO)

    log_and_print("Script scenario started.")

    try:
        scenario = load_scenario()
    except Exception as e:
        log_and_print(f"Error during the loading of scenario. Exception: {e}")
        sys.exit(1)

    if "type" not in scenario:
        log_and_print("Scenario file does not contain type.")
        sys.exit(1)

    try: 
        backend_config = load_from_scenario("backendConfig", None, scenario)
    except Exception as e:
        log_and_print("Error during reading of backend configuration from scenario file")
        sys.exit(1)
    
    if backend_config is None:
        log_and_print("Backend configuration has not been set, values from previous coordinator \
                      runs or the default ones will be used")
        
    else:
        set_backend_config(backend_config)

    # Run backend and wait until it outputs creation of new block filters
    subprocesses_handler = processes_control.Wasabi_Processes_Handler()

    try:
        subprocesses_handler.run_backend()
    except Exception as e:
        log_and_print("An error occured during opening or reading output of the backend subprocess: " + repr(e))
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    # Run client and wait until blocks are downloaded
    try:
        subprocesses_handler.run_client()
    except Exception as e:
        log_and_print("An error occured during opening or reading of client output: " + repr(e))
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    if scenario["type"] == "SimplePassive":
        try:
            scenario_manager = prepare_simple_scenario(scenario)
            print("Returned scenarion manager.")
        except Exception as e:
            log_and_print(f"Error during preparing the simple passive scenario. Error message: {e}")
            subprocesses_handler.clean_subprocesses()
            sys.exit(1)

    # as the wallet does not load the changes in its correspondig file, we have to stop the client and start it again
    # freshly loading all wallets
    try:
        subprocesses_handler.stop_client()
        subprocesses_handler.run_client()
        for wallet in scenario_manager.used_wallets:
            rpc_commands.confirmed_load(wallet)

    except Exception as e:
        subprocesses_handler.clean_subprocesses()
        log_and_print("Error occured during reopening of client app.")


    coinjoin_manager = CoinJoinWalletManager(scenario_manager.used_wallets)
    round_checker = RoundChecker(scenario_manager, coinjoin_manager)

    try:
        round_checker.start()
    except Exception as e:
        log_and_print("While traking scenarios, error occured: " + repr(e))

    log_and_print("Script scenario ended.")
    subprocesses_handler.clean_subprocesses()
