import json
import requests
from time import sleep

def check_txid(txid, ok_txids):
    if txid in ok_txids:
        return True
    
    url = f"https://mempool.space/api/tx/{txid}"
    sleep(0.01)
    try:
        r = requests.get(url)
        print('.', end='')
        if r.status_code == 200:
            return True  # valid transaction
        elif r.status_code == 404:
            return False       # not found / invalid txid
        else:
            print(txid)
            raise Exception(f"Unexpected status {r.status_code}: {r.text}")
    except requests.RequestException as e:
        raise SystemExit(f"Network error: {e}")
    
def filter(txid_map, ok_txids):
    new_txid_map = {}
    c = 0
    for txid in txid_map:
        c += 1
        #print(c)
        if check_txid(txid, ok_txids):
            new_txid_map[txid] = txid_map[txid]
    return new_txid_map

def main(source_file, result_file):
    with open(source_file, "r") as file:
        txid_data = json.load(file)
        ok_txids = txid_data["manual"] | txid_data["crawl_wasabist"] | txid_data["crawl_wabisator"] | txid_data["crawl_crocsapi"]
        old_wabisator = txid_data["crawl_wabisator"]

    db = requests.get("https://wabisator.com/db.json")
    new_wabisator = json.loads(db.text)


    for (coord, data) in new_wabisator.items():
        for tx in data:
            if tx.get("txid") is None or tx["txid"] == "not broadcasted":
                continue
            old_wabisator[tx["txid"]] = coord

    txid_map = filter(old_wabisator, ok_txids)
    print('*', end='')

    txid_data["crawl_wabisator"] = txid_map

    with open(result_file, "w") as file:
        json.dump(txid_data, file, indent=4)


main("../data/wasabi2/txid_coord_new.json", "../data/wasabi2/txid_coord_new.json")



