import helpers.rpc_commands as rpc_commands
import helpers.processes_control as processes_control
import helpers.global_constants as global_constants
import os
from typing import List
import time
import jsonpickle


class CoinWithAnonymity():
    def __init__(self, address, txid, index, wallet, annon_score, amount, spent_in_tx = None):
        self.address = address
        self.txid = txid
        self.index = index
        self.wallet = wallet
        self.annon_score = annon_score
        self.amount = amount
        self.spent_in_tx = spent_in_tx
        
    def as_dict(self):
        return {'address': self.address, 'txid': self.txid, 'index': self.index, 'wallet': self.wallet,
                  'annon_score': self.annon_score, 'amount': self.amount, 'spent_in_tx': self.spent_in_tx}

def parse_wallet_coins(wallet_name: str, coins: dict):
    """
    Creates list of coins with anonymity score for a given wallet from the provided dictionary with coins
    :param wallet_name: name of wallet
    :param coins: dictionary with information returned from Wasabi client RPC
    :return: list of parsed coins
    """
    parsed_coins: List[CoinWithAnonymity] = []
    for coin in coins:
        
        spent_in = None
        if "spentBy" in coin:
            spent_in = coin["spentBy"]
        
        new_coin = CoinWithAnonymity(coin["address"], 
                                     coin["txid"], 
                                     coin["index"], 
                                     wallet_name,
                                     coin["anonymityScore"], 
                                     coin["amount"], 
                                     spent_in)


        parsed_coins.append(new_coin)
    
    return parsed_coins


def list_wallet_coins(wallet: str):
    """
    Creates list of coins with anonymity score for given wallet. Expects that backend and client are running.
    :param wallet: name of wallet
    :return: dictionary with coins retrieved via RPC from the Wasabi wallet client
    """
    rpc_commands.confirmed_load(wallet)
    coins = rpc_commands.list_all_coins(wallet, False)
    coins = coins["result"]

    return parse_wallet_coins(wallet, coins)


def get_coins_for_specified_wallets(wallets: List[str], processes_handler: processes_control.Wasabi_Processes_Handler = None):
    """
    Creates dictionaries indexed by wallet name, address of coins, transaction id and index
    of the coin, and transaction id. Return all 4 dictionaries. Expects, that backend is running.
    :param wallets: list of wallets to load coins from
    :param processes_handler: process handler that will be used to open and close clients
    :return: tuple of fore dictionaries, containing either list of coins (fot by_wallet and by_tx) or
    one coin (by_address and by_tx_id)
    """


    if processes_handler is None:
        processes_handler = processes_control.Wasabi_Processes_Handler()

    by_wallet = {}
    by_address = {}
    by_tx_id = {}
    by_tx = {}
    for wallet in wallets:
        processes_handler.run_client()
        parsed_coins = list_wallet_coins(wallet)
        processes_handler.stop_client()

        by_wallet[wallet] = parsed_coins
        
        for coin in parsed_coins:
            by_address[coin.address] = coin
            by_tx_id[(coin.txid, coin.index)] = coin
            
            if coin.txid not in by_tx:
                by_tx[coin.txid] = []
            
            by_tx[coin.txid].append(coin)
        time.sleep(1)
        

    return by_wallet, by_address, by_tx_id, by_tx


def get_coins_with_starting_backend(wallets: List[str], serialize: bool = False, serialization_path = "serialized_annonymity.json"):
    """
    Creates dictionaries indexed by wallet name, address of coins, transaction id and index
    of the coin, and transaction id. Return all 4 dictionaries. Runs and closes the backend.
    :param wallets: list of wallets to load coins from
    :param serialize: indicator if the list of coins should be serialized
    :param serialization_path: path for the serialization file
    :return: tuple of fore dictionaries, containing either list of coins (fot by_wallet and by_tx) or
    one coin (by_address and by_tx_id)
    """

    process_handler = processes_control.Wasabi_Processes_Handler()
    process_handler.run_backend()

    by_wallet, by_address, by_tx_id, by_tx = get_coins_for_specified_wallets(wallets, process_handler)

    if serialize:
        list_of_coins = list(by_address.values())
        encoded = jsonpickle.encode(list_of_coins)
        with open(serialization_path, "w") as f:
            f.write(encoded)

    process_handler.stop_backend()

    return by_wallet, by_address, by_tx_id, by_tx


def deserialize_to_list(path = "serialized_annonymity.json") -> List[CoinWithAnonymity]:
    """
    Loads list of coins with its annonymity score from json file serialized with jsonpickle and returns list
    of these coins as objects.
    :param path: path for the file with serialized list of coins
    :return: list of CoinWithAnonymity objects deserialized from file
    """

    with open(path, "r") as f:
        
        read_list = f.read()
    
    decoded_coin_list = jsonpickle.decode(read_list)
    if len(decoded_coin_list) > 0:
        print(type(decoded_coin_list[0]))
        if len(decoded_coin_list) > 0 and isinstance(decoded_coin_list[0], dict):
            decoded_coin_list = list(map(
                lambda coin_dict: CoinWithAnonymity(
                    coin_dict["address"],
                    coin_dict["txid"],
                    coin_dict["index"],
                    coin_dict["wallet"],
                    coin_dict["annon_score"],
                    coin_dict["amount"],
                    coin_dict["spent_in_tx"]
                                                ),
                decoded_coin_list
            ))

    return decoded_coin_list

def deserialize_to_dicts(path = "serialized_annonymity.json"):
    """
    Loads list of coins with its annonymity score from json file serialized with jsonpickle and returns dicts
    of these coins as objects.
    :param path: path for the file with serialized list of coins
    :return: tuple of fore dictionaries, containing either list of coins (fot by_wallet and by_tx) or
    one coin (by_address and by_tx_id)
    """

    deser_list = deserialize_to_list(path)
    
    by_wallet = {}
    by_address = {}
    by_tx_id = {}
    by_tx = {}
    
    for coin in deser_list:
        by_address[coin.address] = coin
        by_tx_id[(coin.txid, coin.index)] = coin
        
        if coin.txid not in by_tx:
            by_tx[coin.txid] = [] 
        by_tx[coin.txid].append(coin)

        if coin.wallet not in by_wallet:
            by_wallet[coin.wallet] = []
        by_wallet[coin.wallet].append(coin)

    return by_wallet, by_address, by_tx_id, by_tx


if __name__ == "__main__":
    wallets_availiable = os.listdir(global_constants.GLOBAL_CONSTANTS.path_to_wallets)
    all_wallet_names = list(map(lambda file_name: file_name.split(".")[0], wallets_availiable))
    all_wallet_names.remove(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
    get_coins_with_starting_backend(all_wallet_names, True)