import requests
import time
import os
import json
from typing import List
import logging
from datetime import datetime, timedelta
import Helpers.rpc_commands as rpc_commands
import Helpers.regtest_control as regtest_control
import sys
from dateutil import tz
import Helpers.global_constants as global_constants
import Helpers.processes_control as processes_control
import Helpers.utils as utils
import Helpers.rounds_control as rounds_control


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
        self.input_registration = "InputRegistration"
        self.input_registration_start = translate_json_name("inputRegistrationStart")
        self.input_registration_remaining = translate_json_name("inputRegistrationRemaining")
        self.is_blame_round = translate_json_name("isBlameRound")
        self.id = translate_json_name("id")

backend_names = JsonParamsNaming()


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

    def __init__(self, scenario_manager: rounds_control.ScenarioManager, coin_join_wallet_manager: rounds_control.CoinJoinWalletManager):
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
        utils.log_and_print("Starting the round checker.")
        # time_to_compare_rounds = self.scenario_manager.initial_backend_settings_timestamp + timedelta(seconds=global_constants.GLOBAL_CONSTANTS.config_refresh_time)


        from_setting_backend = datetime.now(tz=tz.tzlocal()) - self.scenario_manager.initial_backend_settings_timestamp 
        # enforcing that new backend configurations were noticed and loaded
        while from_setting_backend.seconds < global_constants.GLOBAL_CONSTANTS.config_refresh_time:
            utils.log_and_print("Configuration for backend was not loaded yet. Waiting for other {} seconds".format(global_constants.GLOBAL_CONSTANTS.config_refresh_time - from_setting_backend.seconds))
            time.sleep(global_constants.GLOBAL_CONSTANTS.config_refresh_time - from_setting_backend.seconds)
            from_setting_backend = datetime.now(tz=tz.tzlocal()) - self.scenario_manager.initial_backend_settings_timestamp

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
                utils.log_and_print("Waiting for round in which inputs can be registered.")
                time.sleep(5)
            else:
                break
        
        self.run()


    def run(self):

        utils.log_and_print(" Started tracking rounds for scenario")
        
        self.coin_join_wallet_manager.start_all()

        while len(self.rounds_finished) < self.round_limit:
            current_rounds = self.get_current_rounds()
            current_rounds_ids = set(map(lambda x: x[backend_names.round_id], current_rounds))
            ended_rounds = []

            for round in self.rounds_active:
                if round[backend_names.round_id] not in current_rounds_ids:
                    utils.log_and_print(" Registered end of round {0}".format(round[backend_names.round_id]))
                    ended_rounds.append(round)
                    self.rounds_finished.append(round)
                    self.rounds_finished_ids.add(round[backend_names.round_id])
                    utils.log_and_print(" This was round number {}".format(self.finished_rounds_tracekd))
                    self.finished_rounds_tracekd += 1


            for round in ended_rounds:
                self.rounds_active.remove(round)
                self.rounds_active_ids.remove(round[backend_names.round_id])
                
                # if round ended, signal it to the manager so it can do work
                self.scenario_manager.progress_round_ended(self.finished_rounds_tracekd)

            
            for round in current_rounds:
                if round[backend_names.round_id] not in self.rounds_active_ids:
                    self.rounds_active.append(round)
                    self.rounds_active_ids.add(round[backend_names.round_id])
                    utils.log_and_print(" Registered creation of round {0}".format(round[backend_names.round_id]))
                    utils.log_and_print(" This is round number {}".format(self.started_rounds_tracked))
                    
                    # when new round is registered, signal it to manager so it can do work
                    self.scenario_manager.progress_round_started(self.started_rounds_tracked)

                    # adding at the end to start indexing by 0
                    self.started_rounds_tracked += 1
            
            if len(ended_rounds) > 0 and global_constants.GLOBAL_CONSTANTS.network == "RegTest":
                regtest_control.mine_block_regtest()
                utils.log_and_print(" Mined new block, because at least one round ended")

            time.sleep(0.5)
        
        self.coin_join_wallet_manager.stop_all()


def load_scenario():
    scenario = {}
    with open("scenario.json", "r") as f:
        scenario = json.load(f)
    return scenario


def load_from_scenario(value, default, scenario):
    if value not in scenario:
        utils.log_and_print("'{0}' setting not specified in scenario, defaulting to {}".format(value, default))
        return default
    return scenario[value]


if __name__ == "__main__":

    logging.basicConfig(filename='automation.log', encoding='utf-8', level=logging.INFO)

    utils.log_and_print("Script scenario started.")

    try:
        scenario = load_scenario()
    except Exception as e:
        utils.log_and_print(f"Error during the loading of scenario. Exception: {e}")
        sys.exit(1)

    if "type" not in scenario:
        utils.log_and_print("Scenario file does not contain type.")
        sys.exit(1)

    try: 
        backend_config = load_from_scenario("backendConfig", None, scenario)
    except Exception as e:
        utils.log_and_print("Error during reading of backend configuration from scenario file")
        sys.exit(1)
    
    if backend_config is None:
        utils.log_and_print("Backend configuration has not been set, values from previous coordinator \
                      runs or the default ones will be used")
        
    else:
        utils.set_backend_config(backend_config)

    backend_set_at = datetime.now(tz= tz.tzlocal())

    # Run backend and wait until it outputs creation of new block filters
    subprocesses_handler = processes_control.Wasabi_Processes_Handler()

    try:
        subprocesses_handler.run_backend()
    except Exception as e:
        utils.log_and_print("An error occured during opening or reading output of the backend subprocess: " + repr(e))
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    # Run client and wait until blocks are downloaded
    try:
        subprocesses_handler.run_client()
    except Exception as e:
        utils.log_and_print("An error occured during opening or reading of client output: " + repr(e))
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    if scenario["type"] == "SimplePassive":
        try:
            scenario_manager = rounds_control.prepare_simple_passive_scenario(scenario)
            print("Returned scenarion manager.")
        except Exception as e:
            utils.log_and_print(f"Error during preparing the simple passive scenario. Error message: {e}")
            subprocesses_handler.clean_subprocesses()
            sys.exit(1)

    elif scenario["type"] == "ComplexPassive":
        try:
            scenario_manager = rounds_control.prepare_complex_passive_scenario(scenario)
            print("Returned scenarion manager.")
        except Exception as e:
            utils.log_and_print(f"Error during preparing the complex passive scenario. Error message: {e}")
            subprocesses_handler.clean_subprocesses()
            sys.exit(1)

    elif scenario["type"] == "SimpleActive":
        try:
            scenario_manager = rounds_control.prepare_simple_active_scenario(scenario)
            print("Returned scenarion manager.")
        except Exception as e:
            utils.log_and_print(f"Error during preparing the simple active scenario. Error message: {e}")
            subprocesses_handler.clean_subprocesses()
            sys.exit(1)
    else:
        utils.log_and_print("Unknown scenario type {}".format(scenario["type"]))
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    # as the wallet does not load the changes in its correspondig file, we have to stop the client and start it again
    # freshly loading all wallets
    try:
        subprocesses_handler.stop_client()

        scenario_manager.set_wallet_configs()

        subprocesses_handler.run_client()
        for wallet in scenario_manager.used_wallets:
            if global_constants.GLOBAL_CONSTANTS.version2:
                rpc_commands.confirmed_load(wallet)
            else:
                rpc_commands.confirmed_select(wallet)

    except Exception as e:
        subprocesses_handler.clean_subprocesses()
        utils.log_and_print("Error occured during reopening of client app.")


    # add information about initial setting of backend configuration, should not be needed, as it is before start of backend
    scenario_manager.initial_backend_settings_timestamp = backend_set_at

    # creates coinjoin manager with wallets
    coinjoin_manager = rounds_control.CoinJoinWalletManager(scenario_manager.used_wallets)

    scenario_manager.wallet_manager = coinjoin_manager

    round_checker = RoundChecker(scenario_manager, coinjoin_manager)

    try:
        round_checker.start()
    except Exception as e:
        utils.log_and_print("While traking scenarios, error occured: " + repr(e))

    utils.log_and_print("Script scenario ended.")
    subprocesses_handler.clean_subprocesses()
