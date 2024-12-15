# Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket coinjoin analysis 

Set of scripts for processing and analysis of datasets created by Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket clients and coordinators. Allows for processing of files extracted by [Dumplings](https://github.com/nopara73/dumplings) tool.  

## Setup
```
git clone https://github.com/crocs-muni/coinjoin-analysis.git
```

## Usage: Parsing Wallet Wasabi 2.x emulations
The scenario assumes previous execution of Wasabi 2.x coinjoins (containerized coordinator and clients) using [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) orchestration tool. After execution, relevant files from containers are serialized as subfolders into ```/path_to_experiments/experiment_1/data/``` folder with the following structure. 
```
  ..
  btc-node           (bitcoin core, regtest blocks)
  wasabi-backend     (wasabi 2.x coordinator container)
  wasabi-client-000  (wasabi 2.x client logs)
  wasabi-client-001
  ...  
  wasabi-client-499
```
Note, that multiple experiments can be stored inside ```/path_to_experiments/``` path. All found folders are checked for ```/data/``` subfolder and if found, the experiment is processed.

### Extraction of coinjoin informaton from original raw files 
To extract all executed coinjoins into unified json format and perform analysis, run:
```
parse_cj_logs.py --action collect_docker --target-path path_to_experiments
```

The extraction process creates the following files: 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, mapping of all wallets to their coins, started rounds, etc.. Used for subsequent analysis.
  * ```wallets_coins.json``` ... information about every output created during execution, mapped to its coinjoin.
  * ```wallets_info.json``` ... information about every address controlled by a given wallet. 

### Re-running analysis from alreday extracted coinjoins 
The coinjoin extraction part is time consuming. If new analysis methods are added or uodated, only the anlaysis part can be re-run. To execute again only analysis (extraction must be already done with files like ```coinjoin_tx_info.json``` already created), run:
```
parse_cj_logs.py --action collect_docker --target-path path_to_experiments
```

If the analysis finishes successfully, the following files are created:
  * ```coinjoin_stats.3.pdf, coinjoin_stats.3.pdf``` ... multiple graphs capturing various analysis results obtained from coinjoin data. 
  * ```coinjoin_tx_info_stats.json``` ... captures information about participation of every wallet in given coinjoin transaction.
  
### Example results
![image](https://github.com/user-attachments/assets/2e5406bc-b8f8-4725-8ff9-6484e805f682)

![image](https://github.com/user-attachments/assets/5325a4ae-468b-4b52-b58f-95d521c15b1c)

