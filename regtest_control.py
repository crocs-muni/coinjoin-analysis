import requests


class RegTest_Btccore_Constants():
    btc_core_pswd = "password"
    btc_core_username = "user"
    bitcoin_regtest_rpc_url = "http://127.0.0.1:18443/"


REGTEST_CONTROL_CONSTANTS = RegTest_Btccore_Constants()


def create_wallet_btc_core(wallet_name: str):
    request = "{\"jsonrpc\": \"2.0\",\"method\": \"createwallet\",\"params\": [\"" + wallet_name + "\"]}"
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url, data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, 
                                                                                            REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    print(response)


def mine_block_regtest(count = 1):
    request = "{\"jsonrpc\": \"2.0\",\"method\": \"getnewaddress\",\"params\": []}"
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url, data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, 
                                                                                          REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    response = response.json()
    print(response)
    address = response["result"]

    print(count)
    request = "{{\"jsonrpc\": \"2.0\",\"method\": \"generatetoaddress\",\"params\": [{0}, \"{1}\"]}}".format(count, address)
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url ,data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    print(response.json())


def get_block_count():
    request = "{\"jsonrpc\": \"2.0\",\"method\": \"getblockcount\",\"params\": []}"
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url, data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, 
                                                                                          REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    response = response.json()
    #print(response["result"])
    return response["result"]


def send_to_address_btc_core(address: str, amount: float):
    request = "{{\"jsonrpc\": \"2.0\",\"method\": \"sendtoaddress\",\"params\": [\"{0}\", {1}]}}".format(address, amount)
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url, data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, 
                                                                                          REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    print(response.json())


def get_btc_balance():
    request = "{\"jsonrpc\": \"2.0\",\"method\": \"getbalance\",\"params\": []}"
    response = requests.post(REGTEST_CONTROL_CONSTANTS.bitcoin_regtest_rpc_url, data = request, auth=(REGTEST_CONTROL_CONSTANTS.btc_core_username, 
                                                                                          REGTEST_CONTROL_CONSTANTS.btc_core_pswd))
    response = response.json()
    result = response["result"]
    return result
    

 