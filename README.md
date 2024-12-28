# Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket coinjoin analysis 

Set of scripts for processing and analysis of datasets created by Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket clients and coordinators. Allows for processing of files extracted by [Dumplings](https://github.com/nopara73/dumplings) tool.  

## Setup
```
git clone https://github.com/crocs-muni/coinjoin-analysis.git
pip install -r requirements.txt
```

## Usage: Parsing Wallet Wasabi 2.x emulations (parse_cj_logs.py)
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

### 1. Extraction of coinjoin informaton from original raw files 
To extract all executed coinjoins into unified json format and perform analysis, run:
```
parse_cj_logs.py --action collect_docker --target-path path_to_experiments
```

The extraction process creates the following files: 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, mapping of all wallets to their coins, started rounds, etc.. Used for subsequent analysis.
  * ```wallets_coins.json``` ... information about every output created during execution, mapped to its coinjoin.
  * ```wallets_info.json``` ... information about every address controlled by a given wallet. 

### 2. Re-running analysis from alreday extracted coinjoins 
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



## Usage: Parsing, analyzing and visualizing mainnet coinjoins from Dumplings (parse_dumplings.py)
This scenario processes data from real coinjoins (Wasabi 1.x, Wasabi 2.x, Whirlpool and others) stored on Bitcoin mainnet, detected and extracted using [Dumplings tool](https://github.com/nopara73/dumplings). 

### 1. Execution of Dumplings tool
See [Dumplings instructions](https://github.com/nopara73/dumplings?tab=readme-ov-file#1-synchronization) for detailed setup and run of the tool.
```
dotnet run -- sync --rpcuser=user --rpcpassword=password
```
After Dumplings tool execution, the relevant files with coinjoin premix, mix and postmix transactions are serialized as plan files into ```/dumplings_output_path``` folder with the following structure:
```
  ..
  Scanner            (Dumplings results, to be processed)
  Stats              (Aggregated Dumplings results, not processed at the moment)
```

### 2. Parsing of Dumplings results into intermediate coinjoin_tx_info.json (```--action process_dumplings```)
To parse coinjoin information from Dumplings files (step 1.) into unified json format (```coinjoin_tx_info.json```) used later for analysis, run:
```
parse_dumplings.py --cjtype ww2 --action process_dumplings --action detect_false_positives --target-path path_to_results
```
The example is given for Wasabi 2.x coinjoins (```--cjtype ww2```). Use ```--cjtype ww1``` for Wasabi 1.x or ```--cjtype sw``` for Samourai Whirlpool instead. 

The extraction process creates the following files into a subfolder of ```Scanner``` named after processed coinjoin protocols (e.g., ```\Scanner\wasabi2\```): 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, etc.. Used for subsequent analysis.
  * ```coinjoin_tx_info_extended.json``` ... additional information extrated about coins and wallets. For real coinjoins, the mapping between coins and wallets is mostly unknown, so this information is separated from ```coinjoin_tx_info.json``` to decrease its size and speedup processing.     
  * ```wasabi2_events.json``` ... Human-readable information about detected coinjoins with most information stripped for readability.
  * ```wasabi2_inputs_distribution.json``` 

Additionally, a subfolder for every month of detected coinjoin activity is created (e.g., ```2022-06-01 00-00-00--2022-07-01 00-00-00...```), cointaining ```coinjoin_tx_info.json``` and ```wasabi2_events.json``` with coinjoin transactions created that specific month for easier handling during analysis later (smaller files). 

Note that based on the coinjoin protocol analyzed, the name of some files may differ. E.g., ```whirlpool_events.json``` for Samourai Whirlpool or ```wasabi1_events.json``` for Wasabi 1.x. 

### 3. Detect and filter false positives (```--action detect_false_positives```)
The Dumplings heuristic coinjoin detection algorithm is not flawless and occasionally selects also transaction, which looks like a coinjoin, but is not. We therefore apply another pass of heuristics to detected like false positives. This step is iterative and requires human interaction to confirm the potential false positives. 

To perform one iteration false positives detection (repeat until no new false positives are found):

1. Run detection (this command utilize already known false positives from ```false_cjtxs.json``` file):
```
parse_dumplings.py --cjtype ww2 --action detect_false_positives --target-path path_to_results
```
2. Inspected created file ```no_remix_txs.json``` with *potential* false positives. 
  - after false positives are confirmed in mempool.space, put them into false_cjtxs.json 
  - 'both_reuse' txs are almost certainly false positives (too many address reuse, default 0.7)
  - 'both' analyze one by one, confirm
  - the typical stop point is when "both", "inputs_address_reuse", "outputs_address_reuse" and "both_reuse" are empty
  - txs left in "inputs" are typically the starting cjtx of some pool
  - txs left in "outputs" are typically the last cjtx of some pool (either pool closed or last mined cjtxs)

3. Repeat whole process again (=> smaller no_remix_txs.json). 
  
Once finished (no new false positives detected), copy ```false_cjtxs.json``` into other folders if multiple pools of the same coinjoin protocol exists (e.g., wasabi2, wasabi2_others, wasabi2_zksnacks)

Note, that false positives are *not* directly removed from ```coinjoin_tx_info.json```. Instead, they are filtered after loading based on the content of ```false_cjtxs.json``` file. As a result, only modification of ```false_cjtxs.json``` is required without change of (large) base files like ```coinjoin_tx_info.json``` and can be quickly recomputed.

### Example results
FIXME
