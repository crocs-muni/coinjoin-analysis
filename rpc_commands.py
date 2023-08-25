import requests
import time


class RPCCommandsConstants():
    client_url = "http://127.0.0.1:37128/"

RPC_COMMANDS_CONSTANTS = RPCCommandsConstants()

def select(walletName, verbose : bool = True):
    select_content = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"selectwallet\", \"params\" : [\"" + walletName + "\", \"pswd\"]}"
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = select_content)
    if verbose:
        print(response.json())
    return response.json()


def start_coinjoin(verbose : bool = True):
    """
    Sends request for starting coinjoin in currently selected wallet.
    :param verbose: specifies if response should be printed
    :return: None
    """

    content_start_coinjoin = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"startcoinjoin\", \"params\":[\"pswd\", true, true]}"
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = content_start_coinjoin)
    if verbose:
        print(response.json()) 


def stop_coinjoin(verbose : bool = True):
    """
    Sends request for stoping coinjoin in currently selected wallet.
    :param verbose: specifies if response should be printed
    :return: None
    """

    content_stop_coinjoin = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"stopcoinjoin\"}"
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = content_stop_coinjoin)
    if verbose:
        print(response.json()) 


def get_wallet_info(verbose : bool = True):
    """
    Sends request for getting information about currently selected wallet.
    :param verbose: specifies if response should be printed
    :return: Client response for request as Response object
    """

    content_get_info = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"getwalletinfo\"}"
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, content_get_info)
    if verbose:
        print(response)
    return response


def get_address(label : str = "redistribution", verbose : bool = True):
    """
    Sends request for getting fresh address of currently selected wallet.
    :param label: label to be added to address
    :param verbose: specifies if response should be printed
    :return: new address as string
    """

    content_get_address = '{"jsonrpc":"2.0","id":"1","method":"getnewaddress","params":["' + label + '"]}'
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = content_get_address)
    
    resp_json = response.json()
    print(resp_json)
    address = resp_json["result"]['address']
    if verbose:
        print(response.json()) 
    return address

def list_unspent(verbose : bool = True):
    """
    Sends request for getting all unspent coins of currently selected wallet.
    :param verbose: specifies if response should be printed
    :return: Client response for request as Response object
    """

    list_content = '{"jsonrpc":"2.0","id":"1","method":"listunspentcoins"}'
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = list_content)
    if verbose:
        print(response.json()) 
    return response.json()

def create_wallet(name : str, pswd : str = "pswd", verbose : bool = True):
    """
    Sends request for creating new wallet. If succesfull, wallet is also connected and selected. Be careful, seed words are currently
    not stored anywhere, if verbose option is turned off and you don't print the response, you will loose the words!
    :param name: name of newly created wallet
    :param pswd: password for newly created wallet
    :param verbose: specifies if response should be printed
    :return: Response to creating new wallet
    """
        
    create_content = '{"jsonrpc":"2.0","id":"1","method":"createwallet","params":["' + name + '", "' + pswd + '"]}'
    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = create_content)
    if verbose:
        print(response.json())
    return response.json()

def confirmed_select(wallet_name : str):
    """
    Sends request for selecting specified wallet. After sending command, blocks until wallet is really selected and loaded.
    :param wallet_name: name of wallet to be loaded
    :return: None
    """

    select(wallet_name, False)
    info_response = get_wallet_info(False)
    json_response = info_response.json()

    while "result" not in json_response or (
            "result" in json_response and
            json_response["result"]["walletName"] != wallet_name):
        # waiting for a second until trying again
        time.sleep(0.5)
        info_response = get_wallet_info(False)
        json_response = info_response.json()


def get_amount_of_coins():
    """
    Returns amount of BTC in selected wallet
    :param wallet_name: name of wallet to be loaded
    :return: number of bitcoins in wallet
    """
    coins = list_unspent(False)["result"]
    amount = 0
    for coin in filter(lambda x: x['confirmed'], coins):
        amount += coin["amount"]
    return amount




# in process of creation if needed, not working yet
def send_to(address, label, verbose : bool = True):
    send_content = '''{"jsonrpc":"2.0","id":"1","method":"send","params": { "payments":[{"sendto": "''' + address + '''", "amount": 8000, "label": "''' + label + '''" }], coins":[{"transactionid":"5637a716d321a08f74227f714d50d4f9ceb70e99a7a17a68146300d3efca8570", "index":1}],"feeTarget":2, "password": "pswd" }}'''

    send_content ='''
    {"jsonrpc":"2.0","id":"1","method":"send", "params": 
    { "payments":[ 
        {"sendto": "''' + "asdfasdasd" + '''", "amount": 2000, "label": "test" }
        ], 
    "coins":[{"transactionid":"5637a716d321a08f74227f714d50d4f9ceb70e99a7a17a68146300d3efca8570", "index":1}], 
    "feeTarget":2, "password": "pswd" }}
    '''

    response = requests.post(RPC_COMMANDS_CONSTANTS.client_url, data = send_content)
    if verbose:
        print(response.json()) 
