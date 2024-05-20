from typing import List, Dict
import logging
import Helpers.utils as utils
import re
import Helpers.rpc_commands as rpc_commands
import Helpers.global_constants as global_constants
import Helpers.regtest_control as regtest_control
import os



class Round():
    def __init__(self, index, backend_config, wallets_configs, active_wallets):
        self.index = index
        self.backend_config = backend_config
        self.wallets_configs = wallets_configs
        self.active_wallets = active_wallets


class RoundWalletChanges():
    def __init__(self, starting_wallets, stopping_wallets):
        self.starting_wallets = starting_wallets
        self.stopping_wallets = stopping_wallets


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


class ScenarioManager():

    wallet_manager: CoinJoinWalletManager = None

    round_count = 0
    participants_count = 0
    starting_funds = []
    fresh_wallets = True

    starting_baceknd_config = None
    starting_wallets_config = None

    used_wallets = []

    initial_backend_settings_timestamp = None

    def progress_round_ended(self, rounds_ended):
        if rounds_ended % 10 == 0:
            if os.path.exists(global_constants.GLOBAL_CONSTANTS.path_to_prison):
                with open(global_constants.GLOBAL_CONSTANTS.path_to_prison, "r") as f:
                    lines = f.readlines()
                utils.log_and_print(f"There are {len(lines)} lines in Prison after round {rounds_ended}")
            else:
                utils.log_and_print(f"There are no lines in Prison after round {rounds_ended}")

    def progress_round_started(self, round_started):
        return
    
    def set_wallet_configs(self):
        return


class SimplePassiveScenarioManager(ScenarioManager):
    def __init__(self, 
                 round_count: int,
                 participants_count: int,
                 backend_config,
                 wallets_config,
                 used_wallets: List[str]):

        self.round_count = round_count
        self.participants_count = participants_count
        self.starting_baceknd_config = backend_config
        self.starting_wallets_config = wallets_config

        self.used_wallets = used_wallets

        self.wallet_manager = CoinJoinWalletManager(used_wallets)

    def progress_round_ended(self, rounds_ended):
        super().progress_round_ended(rounds_ended)

    def progress_round_started(self, round_started):
        super().progress_round_started(round_started)

    def set_wallet_configs(self):
        for wallet in self.used_wallets:
            utils.set_wallet_config(self.starting_wallets_config, wallet)


class ComplexPassiveScenarioManager(SimplePassiveScenarioManager):
    activity_changes_by_round: Dict[int, RoundWalletChanges] = {}
    distinct_wallet_configs = {}

    def __init__(self, 
                 round_count: int,
                 participants_count: int,
                 backend_config,
                 wallets_config,
                 used_wallets: List[str],
                 distinct_wallet_configs,
                 activity_changes_by_round):
        
        super().__init__(round_count, 
                       participants_count, 
                       backend_config, 
                       wallets_config, 
                       used_wallets)
        
        self.distinct_wallet_configs = distinct_wallet_configs
        self.activity_changes_by_round = activity_changes_by_round


    def progress_round_ended(self, rounds_ended):
        super().progress_round_ended(rounds_ended)

        round_active_wallets = self.activity_changes_by_round.get(rounds_ended, None)
        if round_active_wallets is None:
            return
        
        for wallet_to_start in round_active_wallets.starting_wallets:
            self.wallet_manager.start_coinjoin(wallet_to_start)
            utils.log_and_print("Started wallet " + wallet_to_start)

        for wallet_to_stop in round_active_wallets.stopping_wallets:
            self.wallet_manager.stop_coinjoin(wallet_to_stop)
            utils.log_and_print("Stop wallet " + wallet_to_stop)

    def progress_round_started(self, round_started):
        super().progress_round_started(round_started)

    def set_wallet_configs(self):
        utils.log_and_print("Setting configs for all wallets used in the scenario.")
        for index, wallet in enumerate(self.used_wallets):
            if index in self.distinct_wallet_configs:
                utils.set_wallet_config(self.distinct_wallet_configs[index], wallet)
                utils.log_and_print(f"Set specific config for wallet {wallet}.")
            else:
                utils.set_wallet_config(self.starting_wallets_config, wallet)


class SimpleActiveScenarioManager(ComplexPassiveScenarioManager):
    def __init__(self, 
                round_count: int,
                participants_count: int,
                backend_config,
                wallets_config,
                used_wallets: List[str],
                distinct_wallet_configs,
                active_wallets_by_rounds,
                rounds_configs):
    
        super().__init__(round_count, 
                    participants_count, 
                    backend_config, 
                    wallets_config, 
                    used_wallets,
                    distinct_wallet_configs,
                    active_wallets_by_rounds)
    
        self.rounds_configs = rounds_configs


    def progress_round_ended(self, rounds_ended):
        super().progress_round_ended(rounds_ended)


    def progress_round_started(self, round_started):
        super().progress_round_started(round_started)
        
        if round_started in self.rounds_configs:
            utils.set_backend_config(self.rounds_configs[round_started])

    def set_wallet_configs(self):
        # skip the basic one and use the direct parent
        super().set_wallet_configs()


class BasicScenarioSettings():
    def __init__(self,
                 fresh_wallets: bool,
                 starting_funds: List[int],
                 rounds: int,
                 wallet_count: int,
                 default_backend_config,
                 default_wallets_config):
        
        self.fresh_wallets = fresh_wallets
        self.starting_funds = starting_funds
        self.rounds = rounds
        self.wallet_count = wallet_count
        self.default_backend_config = default_backend_config
        self.default_wallets_config = default_wallets_config


class Coin():
    def __init__(self, txid, index, amount):
        self.txid = txid
        self.index = index
        self.amount = amount


def parse_unspent_wallet_coins(trehsold = 5000, wallet: str = global_constants.GLOBAL_CONSTANTS.distributor_wallet_name):
    parsed_coins = []
    if global_constants.GLOBAL_CONSTANTS.version2:
        coins = rpc_commands.list_unspent(wallet, verbose=False)["result"]
    else:
        coins = rpc_commands.list_unspent(verbose=False)["result"]
    for coin in filter(lambda x: x['confirmed'] and x['amount'] > trehsold, coins):
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
    coins = ""
    for index in range(needed_distributor_coins):
        if index != 0:
            coins = coins + ','
        coins = coins + "{{\"transactionid\":\"{0}\", \"index\":{1}}}".format(ordered_distributor_coins[index].txid, 
                                                                         ordered_distributor_coins[index].index)


    coins = "[" + coins + "],"

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
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    else:
        rpc_commands.confirmed_select(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)

    distributor_coins = parse_unspent_wallet_coins(wallet=global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    distribute_coins(distributor_coins, needed_coins)

    if global_constants.GLOBAL_CONSTANTS.network == "RegTest":
        regtest_control.mine_block_regtest()
    else:
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


def load_from_scenario(value, default, scenario):
    if value not in scenario:
        utils.log_and_print("'{0}' setting not specified in scenario, defaulting to {}".format(value, default))
        return default
    return scenario[value]


def load_base(scenario):
        # extracting scenario configurations
    fresh_wallets = load_from_scenario("freshWallets", True, scenario)
    starting_funds = load_from_scenario("startingFunds", [1000000], scenario)
    if not fresh_wallets and "startingFunds" in scenario:
        utils.log_and_print("Starting funds were set but option for creating fresh wallets is turned off, ignoring starting funds option.")
    rounds = load_from_scenario("rounds", 3, scenario)
    participants = load_from_scenario("walletsCounts", 4, scenario)

    backend_config = load_from_scenario("backendConfig", None, scenario)
    if backend_config is None:
        utils.log_and_print("Backend configuration was not specified, using current configuration.")

    wallets_config = load_from_scenario("walletsConfig", None, scenario)
    if wallets_config is None:
        utils.log_and_print("Wallets configuration was not specified, using present configrations of wallets.")

    basic_setting = BasicScenarioSettings(fresh_wallets, starting_funds, rounds, participants, backend_config, wallets_config)

    return basic_setting


def parse_wallets_info(wallets_info, max_allowed_index, parse_configs: bool = False):
    compound_funds = {}
    compound_configs = {}

    for info in wallets_info:
        index = info["walletIndex"]

        if index > max_allowed_index:
            raise RuntimeError(f"Index {index} is larger than number of participants. Max allowed index: {max_allowed_index}.")
        if index in compound_funds or index in compound_configs:
            raise RuntimeError(f"Multiple occurences of index {index} while loading starting wallet infos.")
        
        # if the value contains 
        if "walletFunds" in info: 
            compound_funds[index] = info["walletFunds"]
            print(index, "loading funds - ", compound_funds[index])
        
        # if function is supposed to look for configs, loads them
        if parse_configs and "walletConfig" in info:
            compound_configs[index] = info["walletConfig"]
    
    return compound_funds, compound_configs


def create_n_base_wallets(count, previous_index_number, wallet_name_base = "SimplePassiveWallet"):
    created_wallets = []

    for i in range(count):
        wallet_name = wallet_name_base + repr(previous_index_number + i)
        utils.log_and_print("Creating wallet " + wallet_name)
        rpc_commands.create_wallet(wallet_name)
        created_wallets.append(wallet_name)

    return created_wallets


def get_starting_wallet_number(existing_wallets):
    if len(existing_wallets) > 0:
        last_wallet = existing_wallets[-1]
        numbers = re.findall(r'\d+$', last_wallet)
        if len(numbers) == 0:
            wallet_number = 1
        else:
            wallet_number = int(numbers[-1]) + 1
    else:
        wallet_number = 1
    return wallet_number


def ensure_creation_of_wallets(base_settings: BasicScenarioSettings, 
                               existing_wallets, 
                               wallet_base_name = "SimplePassiveWallet"):

    wallet_number = get_starting_wallet_number(existing_wallets)
    used_wallets = []
    
    # creating needed wallets
    if base_settings.fresh_wallets:
        utils.log_and_print("Creating fresh wallets.")
        
        # if specific settings were provided, allows to set different confgs to wallets

        used_wallets = create_n_base_wallets(base_settings.wallet_count, 
                                                 wallet_number,
                                                 wallet_base_name)

        utils.log_and_print("Created these fresh wallets: " + ", ".join(used_wallets))

    else:
        number_of_existing = len(existing_wallets)
        if number_of_existing >= base_settings.wallet_count:
            used_wallets = existing_wallets[:base_settings.wallet_count]
            utils.log_and_print("Loaded these existing wallets: " + ", ".join(used_wallets))
        else:
            used_wallets = existing_wallets.copy()
            utils.log_and_print("Loaded these existing wallets: " + ", ".join(used_wallets))
            utils.log_and_print(f"Creating {base_settings.wallet_count - number_of_existing} new wallets.")


            new_wallets_created = create_n_base_wallets(base_settings.wallet_count - number_of_existing, 
                                                            wallet_number, 
                                                            wallet_base_name)
            
            used_wallets.extend(new_wallets_created)
    
    return used_wallets


def parse_wallets_in_rounds(rounds_parameters, wallets):
    rounds = {}

    max_allowed_index = len(wallets) - 1

    for round in rounds_parameters:

        index = round["index"]

        if index in rounds:
            raise RuntimeError(f"Noticed duplicate index: {index} when loading wallets activity")
        
        starting_wallets = set(round.get("startWallets", []))
        stoping_wallets = set(round.get("stopWallets", []))

        if len(starting_wallets) > 0 and (max(starting_wallets) > max_allowed_index or min(starting_wallets) < 0):
            raise RuntimeError(f"There are indices that are greater then the number of wallet in selection of starting wallets for round {index}")
        
        if len(stoping_wallets) > 0 and (max(stoping_wallets) > max_allowed_index or min(stoping_wallets) < 0):
            raise RuntimeError(f"There are indices that are greater then the number of wallet in selection of stopping wallets for round {index}")

        if len(starting_wallets.intersection(stoping_wallets)) > 0:
            raise RuntimeError(f"There is mismatch in activating and stoping coinjoin. Some wallets are supposed to be \
                               stopped and also started in round {index}")


        rounds[index] = (starting_wallets, stoping_wallets)
    
    activity_changes = {}

    # presumption that in the first round, all wallets are active
    active_wallets = set([x for x in range(len(wallets))])
    stoped_wallets = set()

    # transform indices to names. Can not be done in previous for loop as the indices of round can be not sorted
    for round_index in sorted(rounds):
        
        config_start_indices, config_stop_indices = rounds[round_index]

        # create intersection with opposite status of wallets from previous round - to prevent not necessery loading of wallets
        to_start_indices = stoped_wallets.intersection(config_start_indices)
        to_stop_indices = active_wallets.intersection(config_stop_indices)

        to_start_names = [wallets[wallet_index] for wallet_index in to_start_indices]
        to_end_names = [wallets[wallet_index] for wallet_index in to_stop_indices]

        activity_changes[round_index] = RoundWalletChanges(to_start_names, to_end_names)

        # update active wallets for current round
        active_wallets = active_wallets.difference(to_stop_indices)
        active_wallets = active_wallets.union(to_start_indices)

        # update stopped wallets for current round
        stoped_wallets = stoped_wallets.difference(to_start_indices)
        stoped_wallets = stoped_wallets.union(to_stop_indices)

        if len(active_wallets.intersection(stoped_wallets)) > 0:
            raise RuntimeError("Error during parsing the wallets activity changes in rounds")

    return activity_changes


def parse_round_configs(rounds_config):
    indexed_round_configs = {}

    for round in rounds_config:
        index = round["index"]
        if index in indexed_round_configs:
            raise RuntimeError(f"Collision in index {index} for round configruations.")
        
        if "backendConfig" in round:
            indexed_round_configs[index] = round["backendConfig"]
    return indexed_round_configs


def orchestrate_funds_distribution(base_settings, used_wallets, compound_funds):   
    if base_settings.fresh_wallets:
        print("Started funds redistribution")
        prepare_wallets_values(used_wallets, compound_funds, base_settings.starting_funds)
        print("Coins redistribution finished.")
    else:
        prepare_wallets_amount(used_wallets, base_settings.rounds * 5000)


def prepare_simple_passive_scenario(scenario):
    base_settings = load_base(scenario)

    wallets_info = load_from_scenario("walletsInfo", [], scenario)
    compound_funds = {}
    if len(wallets_info) > 0:
        compound_funds, compound_configs = parse_wallets_info(wallets_info, base_settings.wallet_count - 1)
    
    print(compound_funds)

    existing_wallets = list(filter(lambda x: "SimplePassiveWallet" in x, utils.WalletsListings().wallets_availiable))
    # sorting by number, that is after 'SimplePassiveWallet' text
    existing_wallets.sort(key= lambda x: int(x[19:]))

    used_wallets = ensure_creation_of_wallets(base_settings, existing_wallets)

    # loading all wallets used in the scenario
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
        for wallet in used_wallets:
            rpc_commands.confirmed_load(wallet)

    orchestrate_funds_distribution(base_settings, used_wallets, compound_funds)
    
    scenario_manager = SimplePassiveScenarioManager(
                                                base_settings.rounds,
                                                base_settings.wallet_count,
                                                base_settings.default_backend_config,
                                                base_settings.default_wallets_config,
                                                used_wallets)
    
    return scenario_manager


def prepare_complex_passive_scenario(scenario):
    base_settings = load_base(scenario)

    wallets_info = load_from_scenario("walletsInfo", [], scenario)

    compound_funds = {}
    compound_configs = {}
    if len(wallets_info) > 0:
        compound_funds, compound_configs = parse_wallets_info(wallets_info, base_settings.wallet_count - 1, True)
    
    existing_wallets = list(filter(lambda x: "AdvancedPassiveWallet" in x, utils.WalletsListings().wallets_availiable))
    # sorting by number, that is after 'SimplePassiveWallet' text
    existing_wallets.sort(key= lambda x: int(x[21:]))

    used_wallets = ensure_creation_of_wallets(base_settings, existing_wallets, "AdvancedPassiveWallet")

    # loading all wallets used in the scenario
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
        for wallet in used_wallets:
            rpc_commands.confirmed_load(wallet)


    orchestrate_funds_distribution(base_settings, used_wallets, compound_funds)

    wallets_activity_settings = load_from_scenario("roundsConfigs", [], scenario)

    activity_changes = parse_wallets_in_rounds(wallets_activity_settings, used_wallets)

    scenario_manager = ComplexPassiveScenarioManager(
                                                    base_settings.rounds,
                                                    base_settings.wallet_count,
                                                    base_settings.default_backend_config,
                                                    base_settings.default_wallets_config,
                                                    used_wallets,
                                                    compound_configs,
                                                    activity_changes)
    
    return scenario_manager


def prepare_simple_active_scenario(scenario):
    base_settings = load_base(scenario)

    wallets_info = load_from_scenario("walletsInfo", [], scenario)

    compound_funds = {}
    compound_configs = {}
    if len(wallets_info) > 0:
        compound_funds, compound_configs = parse_wallets_info(wallets_info, base_settings.wallet_count - 1, True)
    
    existing_wallets = list(filter(lambda x: "SimpleActiveWallet" in x, utils.WalletsListings().wallets_availiable))
    # sorting by number, that is after 'SimplePassiveWallet' text
    existing_wallets.sort(key= lambda x: int(x[18:]))

    used_wallets = ensure_creation_of_wallets(base_settings, existing_wallets, "SimpleActiveWallet")

    # loading all wallets used in the scenario
    if global_constants.GLOBAL_CONSTANTS.version2:
        rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
        for wallet in used_wallets:
            rpc_commands.confirmed_load(wallet)


    orchestrate_funds_distribution(base_settings, used_wallets, compound_funds)

    wallets_activity_settings = load_from_scenario("roundsConfigs", [], scenario)
    
    activity_changes = parse_wallets_in_rounds(wallets_activity_settings, used_wallets)

    loaded_rounds = load_from_scenario("roundsConfigs", [], scenario)

    rounds_configs = parse_round_configs(loaded_rounds)


    scenario_manager = SimpleActiveScenarioManager(
                                                    base_settings.rounds,
                                                    base_settings.wallet_count,
                                                    base_settings.default_backend_config,
                                                    base_settings.default_wallets_config,
                                                    used_wallets,
                                                    compound_configs,
                                                    activity_changes,
                                                    rounds_configs)
    
    return scenario_manager
