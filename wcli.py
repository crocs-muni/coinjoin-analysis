# wcli - WasabiWallet command line interpreter 
#
# Adapted from https://github.com/zkSNACKs/WalletWasabi/blob/master/Contrib/CLI/wcli.sh using ChatGPT and manual code fixes (Petr Svenda)
#
# Copyright (c) 2023 zkSNACKs, Petr Svenda (@petrs)
#
# MIT license.
import os.path
import subprocess
import json
import sys

WASABIWALLET_DATA_DIR = ""
LEGACY_API = True
VERBOSE = True


def config_extract(query):
    config_path = os.path.join(WASABIWALLET_DATA_DIR, "Client", "Config.json")
    result = subprocess.run(["jq", "-r", query, config_path], capture_output=True, text=True, shell=True)
    return result.stdout.strip()


def wcli(args):
    credentials = config_extract('.JsonRpcUser + ":" + .JsonRpcPassword')
    endpoint = config_extract('.JsonRpcServerPrefixes[0]')
    basic_auth = "" if credentials == ":" else "-u " + credentials

    wallet_name = ""

    if not LEGACY_API:
        if args and args[0].startswith("-wallet="):
            wallet_name = args[0][8:] + "/"
            args = args[1:]

    method = args[0]
    params = args[1:]

    request_data = {
        "jsonrpc": "2.0",
        "id": "curltext",
        "method": method,
        "params": params
    }
    request = json.dumps(request_data)
    request = request.replace("\"", "\\\"")
    curl_command = [
        "curl", "-s", basic_auth, "--data-binary", '\"', request, '\"',
        "-H", repr("content-type: text/plain;"), repr(endpoint), "-v"
    ]
    curl_command_str = ' '.join(s for s in curl_command)

    curl_command_str = curl_command_str.replace('\'', "\"")
    if VERBOSE:
        print(curl_command_str)

    #curl_command_str_getwalletinfo = "curl -u test:pswd --data-binary \"{\"jsonrpc\": \"2.0\", \"id\": \"curltext\", \"method\": \"getwalletinfo\", \"params\": []}\" -H \"content-type: text/plain;\" \"http://127.0.0.1:37128/Wallet1/\" -v"
    #                    curl -u test:pswd --data-binary "{\"jsonrpc\": \"2.0\", \"id\": \"curltext\", \"method\": \"getwalletinfo\", \"params\": []}" -H "content-type: text/plain;" "http://127.0.0.1:37128/Wallet1/" -v
    #curl_command_str2 = "curl -u test:pswd --data-binary \"{\"jsonrpc\": \"2.0\", \"id\": \"curltext\", \"method\": \"getwalletinfo\", \"params\": []}\" -H \"content-type: text/plain;\" \"http://127.0.0.1:37128/Wallet1/\" -v"
    #print(curl_command_str2)
    #result = subprocess.run(curl_command_str_getwalletinfo, capture_output=True, text=True)

    result = subprocess.run(curl_command_str, capture_output=True, text=True)

    curl_errorcode = result.returncode
    result_output = result.stdout.strip()
    result_error = json.loads(result_output).get("error")

    curl_fail_to_connect_errorcode = 7

    rawprint = ["help"]
    result_json = ""
    if curl_errorcode == curl_fail_to_connect_errorcode:
        if VERBOSE:
            print("It was not possible to get a response. RPC server could be disabled.")
        result_json = None
    elif result_error is None:
        output = json.loads(result_output)
        if method in rawprint:
            if "result" in output.keys():
                if VERBOSE:
                    print(output["result"])
            else:
                if VERBOSE:
                    print("OK: No result returned, printing whole response json: " + json.dumps(output))
        else:
            if "result" not in output.keys():
                if VERBOSE:
                    print("OK: No result returned, printing whole response json: " + json.dumps(output))
            else:
                result_json = json.loads(result_output)["result"]
                if isinstance(result_json, list) and len(result_json) > 0:
                    keys = "\t".join(result_json[0].keys())
                    values = "\n".join("\t".join(map(str, item.values())) for item in result_json)
                    if VERBOSE:
                        print(keys + "\n" + values)
                else:
                    if VERBOSE:
                        print(result_json)
    else:
        print(result_error["message"])
        result_json = None

    return result_json
