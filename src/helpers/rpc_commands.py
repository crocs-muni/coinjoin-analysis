import os

import requests
import time
import json
import helpers.global_constants as global_constants


class RPCCommandsConstants():
    client_url = "http://127.0.0.1:37128/"
    wasabi_wallet_data = "c:\\Users\\xsvenda\\AppData\\Roaming\\WalletWasabi\\"
    rpc_user = ""
    rpc_pswd = ""

    def __init__(self):
        self.config_path = "{}/Client/Config.json".format(self.wasabi_wallet_data)
        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                loaded_config = json.load(f)
                self.rpc_user = loaded_config["JsonRpcUser"]
                self.rpc_pswd = loaded_config["JsonRpcPassword"]
        else:
            print('WARNING: {} not found'.format(self.config_path))


RPC_COMMANDS_CONSTANTS = RPCCommandsConstants()



def send_post(data,
              specified_wallet: str = None,
              different_endpoint: str = None):
    """
    Wraped post. Allows authentication, if rpc credentials are provided.
    :param data: data part of post request
    :param specified_wallet: name of the wallet. Newer wasabi versions
    changed rpc comunication s.t.
    wallet names are used dirrectly in endpoint
    :param different_endpoint: allows users to specify different endpoint
    as string. May be useful for the dockerized version.
    """
    if different_endpoint is not None:
        efective_endpoint = different_endpoint
    else:
        efective_endpoint = global_constants.GLOBAL_CONSTANTS.client_endpoint

    if efective_endpoint[-1] != "/":
        efective_endpoint = efective_endpoint + "/"

    if specified_wallet is not None:
        efective_endpoint = efective_endpoint + specified_wallet

    if global_constants.GLOBAL_CONSTANTS.rpc_pswd != "" and global_constants.GLOBAL_CONSTANTS.rpc_user != "":
        response = requests.post(efective_endpoint,
                                 data=data,
                                 auth=(global_constants.GLOBAL_CONSTANTS.rpc_user,
                                       global_constants.GLOBAL_CONSTANTS.rpc_pswd))
    else:
        response = requests.post(efective_endpoint, data=data)
    return response


def select(wallet_name, verbose: bool = True):
    """
    method is no longer used in RPC in version 2.0.4!!"
    """
    select_content = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"selectwallet\", \"params\" : [\"" + wallet_name + "\", \"pswd\"]}"
    response = send_post(select_content)
    if verbose:
        print(response.json())
    return response.json()


def start_coinjoin(wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for starting coinjoin in currently selected wallet.
    :param wallet: name of the wallet that should start coinjoin
    :param verbose: specifies if response should be printed
    :return: None
    """

    content_start_coinjoin = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"startcoinjoin\", \"params\":[\"pswd\", true, true]}"
    response = send_post(content_start_coinjoin, wallet, different_endpoint)
    if verbose:
        print(response.json())


def stop_coinjoin(wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for stoping coinjoin in currently selected wallet.
    :param wallet: name of the wallet that should start coinjoin
    :param verbose: specifies if response should be printed
    :return: None
    """

    content_stop_coinjoin = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"stopcoinjoin\"}"
    response = send_post(content_stop_coinjoin, wallet, different_endpoint)
    if verbose:
        print(response.json())


def get_wallet_info(wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for getting information about currently selected wallet.
    :param wallet: name of the wallet that should stop coinjoin
    :param verbose: specifies if response should be printed
    :return: Client response for request as Response object
    """

    content_get_info = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"getwalletinfo\"}"
    response = send_post(content_get_info, wallet, different_endpoint)
    if verbose:
        print(response)
    return response


def get_address(label: str = "redistribution", wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for getting fresh address of currently selected wallet.
    :param label: label to be added to address
    :param wallet: name of the wallet for which new address should be returned
    :param verbose: specifies if response should be printed
    :return: new address as string
    """

    content_get_address = '{"jsonrpc":"2.0","id":"1","method":"getnewaddress","params":["' + label + '"]}'
    response = send_post(content_get_address, wallet, different_endpoint)

    resp_json = response.json()
    print(resp_json)
    address = resp_json["result"]['address']
    if verbose:
        print(response.json())
    return address


def list_unspent(wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for getting all unspent coins of currently selected wallet.
    :param wallet: name of the wallet that for listing its coins
    :param verbose: specifies if response should be printed
    :return: Client response for request as Response object
    """

    list_content = '{"jsonrpc":"2.0","id":"1","method":"listunspentcoins"}'
    response = send_post(list_content, wallet, different_endpoint)
    if verbose:
        print(response.json())
    return response.json()


def list_all_coins(wallet: str = None, verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for getting all coins (including those from past) of chosen wallet.
    :param wallet: name of the wallet that for listing its coins
    :param verbose: specifies if response should be printed
    :return: Client response for request as Response object
    """

    list_content = '{"jsonrpc":"2.0","id":"1","method":"listcoins"}'
    response = send_post(list_content, wallet, different_endpoint)
    if verbose:
        print(response.json())
    return response.json()


def create_wallet(name: str, pswd: str = "pswd", verbose: bool = True, different_endpoint: str = None):
    """
    Sends request for creating new wallet. If succesfull, wallet is also
    connected and selected. Be careful, seed words are currently not stored
    anywhere, if verbose option is turned off and you don't print the response,
    you will loose the words!
    :param name: name of newly created wallet
    :param pswd: password for newly created wallet
    :param verbose: specifies if response should be printed
    :return: Response to creating new wallet
    """

    create_content = '{"jsonrpc":"2.0","id":"1","method":"createwallet","params":["' + name + '", "' + pswd + '"]}'
    response = send_post(create_content, different_endpoint=different_endpoint)
    if verbose:
        print(response.json())
    return response.json()


def confirmed_select(wallet_name: str):
    """
    !!!! No longer works in new version of app!!!
    Sends request for selecting specified wallet. After sending
    command, blocks until wallet is really selected and loaded.
    :param wallet_name: name of wallet to be loaded
    :return: None
    """

    select(wallet_name, False)
    info_response = get_wallet_info(verbose=False)
    json_response = info_response.json()

    while "result" not in json_response or (
            "result" in json_response and
            json_response["result"]["walletName"] != wallet_name):
        # waiting for a second until trying again
        time.sleep(0.5)
        info_response = get_wallet_info(verbose=False)
        json_response = info_response.json()


def confirmed_load(wallet_name: str, different_endpoint: str = None):
    """
    Alternative for confirmed select for version 2.0.4. It is not needed
    to select wallets, but they still has to be loaded, which takes time
    for the first time.
    :param wallet_name: string name of wallet to be loaded
    """

    response = get_wallet_info(wallet_name, verbose=False, different_endpoint = different_endpoint)
    json_response = response.json()
    while "error" in json_response:
        time.sleep(0.2)
        response = get_wallet_info(wallet_name, verbose=False, different_endpoint = different_endpoint)
        json_response = response.json()


def get_amount_of_coins(wallet: str = None, different_endpoint: str = None):
    """
    Returns amount of BTC in selected wallet
    :param wallet: name of the wallet that for counting the coins
    :param wallet_name: name of wallet to be loaded
    :return: number of bitcoins in wallet
    """
    coins = list_unspent(wallet, verbose=False, different_endpoint = different_endpoint)["result"]
    amount = 0
    for coin in filter(lambda x: x['confirmed'], coins):
        amount += coin["amount"]
    return amount
