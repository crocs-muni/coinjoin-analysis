# Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket coinjoin analysis 

Set of scripts for processing and analysis of datasets created by Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket clients and coordinators. Allows for processing of Bitcoin mainnet coinjoins as extracted by [Dumplings](https://github.com/nopara73/dumplings) tool.  

## Setup
```
git clone https://github.com/crocs-muni/coinjoin-analysis.git
pip install -r requirements.txt
```

## Supported operations

1. [Process mainnet coinjoins collected by Dumplings (```parse_dumplings.py```)](#process-dumplings)
    1. [Execute Dumplings tool](#run-dumplings)
    1. [Parse Dumplings results into intermediate coinjoin_tx_info.json (```--action process_dumplings```)](#process-dumplings)
    1. [Detect and filter false positives (```--action detect_false_positives```)](#detect-false-positives)
    1. [Analyze and plot results (```--action plot_coinjoins```)](#plot-coinjoins)
    1. [Example results](#dumplings-examples)
1. [Process Wallet Wasabi 2.x emulations from EmuCoinJoin (```parse_cj_logs.py```)](#ecj-process)
    1. [Execute EmuCoinJoin emulator](#run-ecj)
    1. [Extract coinjoin information from original raw files (```--action collect_docker```)](#ecj-extract)
    1. [Re-run analysis from alreday extracted coinjoins (```--action analyze_only```)](#ecj-rerun)
    1. [Example results](#ecj-examples)
---

<a id="process-dumplings"></a>
## Usage: Parse, analyze and visualize mainnet coinjoins from Dumplings (```parse_dumplings.py```)
This scenario processes data from real coinjoins (Wasabi 1.x, Wasabi 2.x, Whirlpool and others) stored on Bitcoin mainnet, detected and extracted using [Dumplings tool](https://github.com/nopara73/dumplings). 

<a id="run-dumplings"></a>
### 1. Execute Dumplings tool
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

<a id="process-dumplings"></a>
### 2. Parse Dumplings results into intermediate coinjoin_tx_info.json (```--action process_dumplings```)
To parse coinjoin information from Dumplings files (step 1.) into unified json format (```coinjoin_tx_info.json```) used later for analysis, run:
```
parse_dumplings.py --cjtype ww2 --action process_dumplings --target-path path_to_results
```
The example is given for Wasabi 2.x coinjoins (```--cjtype ww2```). Use ```--cjtype ww1``` for Wasabi 1.x or ```--cjtype sw``` for Samourai Whirlpool instead. 

The extraction process creates the following files into a subfolder of ```Scanner``` named after processed coinjoin protocols (e.g., ```\Scanner\wasabi2\```): 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, etc.. Used for subsequent analysis.
  * ```coinjoin_tx_info_extended.json``` ... additional information extrated about coins and wallets. For real coinjoins, the mapping between coins and wallets is mostly unknown, so this information is separated from ```coinjoin_tx_info.json``` to decrease its size and speedup processing.     
  * ```wasabi2_events.json``` ... Human-readable information about detected coinjoins with most information stripped for readability.
  * ```wasabi2_inputs_distribution.json``` 

Additionally, a subfolder for every month of detected coinjoin activity is created (e.g., ```2022-06-01 00-00-00--2022-07-01 00-00-00...```), cointaining ```coinjoin_tx_info.json``` and ```wasabi2_events.json``` with coinjoin transactions created that specific month for easier handling during analysis later (smaller files). 

Note that based on the coinjoin protocol analyzed, the name of some files may differ. E.g., ```whirlpool_events.json``` for Samourai Whirlpool or ```wasabi1_events.json``` for Wasabi 1.x. 

<a id="detect-false-positives"></a>
### 3. Detect and filter false positives (```--action detect_false_positives```)
The Dumplings heuristic coinjoin detection algorithm is not flawless and occasionally selects also transaction, which looks like a coinjoin, but is not. We therefore apply another pass of heuristics to detected like false positives. This step is iterative and requires human interaction to confirm the potential false positives. 

The detection in each iteration utilizes already known false positives loaded from ```false_cjtxs.json``` file. You may download pre-prepared files for different coinjoin protocols already manually filtered by us here (file commit date corresponds approximately to ):
  - Wasabi 1.x: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/wasabi1/false_cjtxs.json)  (last coinjoin 2024-05-30)
  - Wasabi 2.x: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/wasabi2/false_cjtxs.json)  (new coinjoins still created, needs update)
  - Whirlpool: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/whirlpool/false_cjtxs.json) (last coinjoin 2024-04-25, empty file, no false positives by Dumplings)

To perform one iteration false positives detection (repeat until no new false positives are found):

#### 3.1. Run detection (this command utilize already known false positives from ```false_cjtxs.json``` file):
```
parse_dumplings.py --cjtype ww2 --action detect_false_positives --target-path path_to_results
```
#### 3.2. Inspect created file ```no_remix_txs.json``` containing *potential* false positives. 
Here are some tips for detection of false positives:
  - 'both_reuse_0_70' txs are almost certainly false positives (too many address reused, default threshold is 70% of reused addresses, normal coinjoins are having almost all addresses freshly generated). Put them all into false_cjtxs.json and rerun.
  - 'both_noremix' txs are transactions with no input and no output conected to other known coinjoin transaction. Very likely false positive, but needs to be analyzed one by one to confirm. 
  - txs left in "inputs_noremix" after all are typically the starting cjtx of some pool (no previous coinjoin was executed).
  - txs left in "outputs_noremix" are typically the last cjtx of some pool (either pool closed and no longer produce transactions, or is last mined cjtx(s) wrt Dumpling sync date)
  - after false positives are confirmed in mempool.space, put them into false_cjtxs.json 

#### 3.3. Repeat whole process again (=> smaller no_remix_txs.json). 
  - the typical stop point is when "both", "inputs_address_reuse", "outputs_address_reuse" and "both_reuse" are empty
  
Once finished (no new false positives detected), copy ```false_cjtxs.json``` into other folders if multiple pools of the same coinjoin protocol exists (e.g., wasabi2, wasabi2_others, wasabi2_zksnacks)

Note, that false positives are *not* directly removed from ```coinjoin_tx_info.json```. Instead, they are filtered after loading based on the content of ```false_cjtxs.json``` file. As a result, only modification of ```false_cjtxs.json``` is required without change of (large) base files like ```coinjoin_tx_info.json``` and can be quickly recomputed.

<a id="plot-coinjoins"></a>
### 4. Analyze and plot results (```--action plot_coinjoins```)
To analyze and plot various analysis graphs from processed coinjoins, run:
```
parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path path_to_results
```
This command generates several files with analysis and visualization of executed coinjoins. For visualizations, both png and pdf file formats are generated - use *.pdf where necessary as not all details may be visible in larger *.png files. 

The files are named using the following convention: 
  - ```_values_``` means visualization of values of coinjoin inputs  
  - ```_nums_``` means visualization of number of coinjoin inputs  
  - ```_norm_``` means normalization of values before analysis  
  - ```_notnorm_``` means no normalization is performed before analysis  

The following files are generated:
  - ```*_remixrate_[values/nums]_[norm/notnorm].json``` cointains remix rate 
  - ```*_input_types_values_notnorm.pdf```

<a id="dumplings-examples"></a>
### 5. Example results
Vizualized liquidity changes in Wasabi 1.x, Wasabi 2.x and Whirlpool coinjoins 
![image](https://github.com/user-attachments/assets/33af36a6-8650-47dc-b92a-f5c611962b72)

Value of Wasabi 2.x coinjoin inputs during December 2023: 
![image](https://github.com/user-attachments/assets/9d327604-b0e5-4c60-86c0-c6e04f01b694)

Value of Wasabi 2.x coinjoin inputs during December 2023 (normalized): 
![image](https://github.com/user-attachments/assets/2364a7ce-e45b-48ac-825c-aeb755a65dfd)

Number of Wasabi 2.x coinjoin inputs during first month of operation: 
![image](https://github.com/user-attachments/assets/6362d74f-7c5a-4020-9e85-0c41e991c263)

Number of Wasabi 2.x coinjoin inputs during first month of operation (normalized): 
![image](https://github.com/user-attachments/assets/a18c9d91-e416-48e3-bb7c-e39883bc6c5b)

Value of Wasabi 2.x coinjoins for post-zkSNACKS coordinators (June-December 2024): 
![image](https://github.com/user-attachments/assets/69ebb029-83f0-493c-bbb7-11b9b86fd746)

---

<a id="ecj-process"></a>
## Usage: Parse Wallet Wasabi 2.x emulations from EmuCoinJoin (```parse_cj_logs.py```)
The scenario assumes previous execution of Wasabi 2.x and JoinMarket coinjoins (produced by containerized coordinator and clients) using [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) orchestration tool. 

<a id="run-ecj"></a>
### 1. Execute EmuCoinJoin emulator
See [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) for detailed setup and run of the tool.
After EmuCoinJoin execution, relevant files from containers are serialized as subfolders into ```/path_to_experiments/experiment_1/data/``` folder with the following structure. 
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

<a id="ecj-extract"></a>
### 2. Extract coinjoin information from original raw files (```--action collect_docker```)
To extract all executed coinjoins into unified json format and perform analysis, run:
```
parse_cj_logs.py --action collect_docker --target-path path_to_experiments
```

The extraction process creates the following files: 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, mapping of all wallets to their coins, started rounds, etc.. Used for subsequent analysis.
  * ```wallets_coins.json``` ... information about every output created during execution, mapped to its coinjoin.
  * ```wallets_info.json``` ... information about every address controlled by a given wallet. 

<a id="ecj-rerun"></a>
### 3. Re-run analysis from already extracted coinjoins (```--action analyze_only```)
The coinjoin extraction part is time consuming. If new analysis methods are added or udated, only the anlaysis part can be re-run. To execute again only analysis (extraction must be already done with files like ```coinjoin_tx_info.json``` already created), run:
```
parse_cj_logs.py --action analyze_only --target-path path_to_experiments
```

If the analysis finishes successfully, the following files are created:
  * ```coinjoin_stats.3.pdf, coinjoin_stats.3.pdf``` ... multiple graphs capturing various analysis results obtained from coinjoin data. 
  * ```coinjoin_tx_info_stats.json``` ... captures information about participation of every wallet in given coinjoin transaction.

<a id="ecj-examples"></a>
### 4. Example results
![image](https://github.com/user-attachments/assets/2e5406bc-b8f8-4725-8ff9-6484e805f682)

![image](https://github.com/user-attachments/assets/5325a4ae-468b-4b52-b58f-95d521c15b1c)

---
