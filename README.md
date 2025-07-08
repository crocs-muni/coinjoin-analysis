# Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket coinjoin analysis 

Set of scripts for processing, analysis, and visualization of coinjoin transactions. Performs processing and visualization of 1) real coinjoins as extracted from Bitcoin mainnet by [Dumplings](https://github.com/nopara73/dumplings) tool (no ground truth knowledge about coins to wallets mapping) and 2) base files with coinjoins for  Wallet Wasabi 1.x, Wallet Wasabi 2.x and JoinMarket clients and coordinators executed in emulated environment by [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) (known mapping between coins and wallets). 

## Setup
Clone repository:
```
git clone https://github.com/crocs-muni/coinjoin-analysis.git
```

Optional: make Python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```
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
    1. [Re-run analysis from already extracted coinjoins (```--action analyze_only```)](#ecj-rerun)
    1. [Example results](#ecj-examples)
---

<a id="process-dumplings"></a>
## Usage: Parse, analyze, and visualize mainnet coinjoins from Dumplings (```parse_dumplings.py```)
This usage scenario processes data from real coinjoins (Wasabi 1.x, Wasabi 2.x, Whirlpool, and others) stored on the Bitcoin mainnet as detected and extracted using [Dumplings tool](https://github.com/nopara73/dumplings). 

<a id="run-dumplings"></a>
### 1. Execute Dumplings tool
See [Dumplings instructions](https://github.com/nopara73/dumplings?tab=readme-ov-file#1-synchronization) for detailed setup and run of the tool.
```
dotnet run --sync --rpcuser=user --rpcpassword=password
```
After Dumplings tool execution, the relevant files with coinjoin premix, mix, and postmix transactions are serialized as plan files into ```/dumplings_output_path``` folder with the following structure:
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
  * ```coinjoin_tx_info_extended.json``` ... additional information extracted about coins and wallets. For real coinjoins, the mapping between coins and wallets is mostly unknown, so this information is separated from ```coinjoin_tx_info.json``` to decrease its size and speed up processing.     
  * ```wasabi2_events.json``` ... Human-readable information about detected coinjoins with most information stripped for readability.
  * ```wasabi2_inputs_distribution.json``` 

Additionally, a subfolder for every month of detected coinjoin activity is created (e.g., ```2022-06-01 00-00-00--2022-07-01 00-00-00...```), containing ```coinjoin_tx_info.json``` and ```wasabi2_events.json``` with coinjoin transactions created that specific month for easier handling during analysis later (smaller files). 

Note that based on the coinjoin protocol analyzed, the name of some files may differ. E.g., ```whirlpool_events.json``` for Samourai Whirlpool or ```wasabi1_events.json``` for Wasabi 1.x. 

<a id="detect-false-positives"></a>
### 3. Detect and filter false positives (```--action detect_false_positives```)
The Dumplings heuristic coinjoin detection algorithm is not flawless and occasionally selects a transaction that looks like a coinjoin but is not. We, therefore, apply another pass of heuristics to detect such false positives. This step is iterative and requires human interaction to confirm the potential false positives. Note that false positives are *not* directly removed from ```coinjoin_tx_info.json```. Instead, they are filtered after loading based on the content of ```false_cjtxs.json``` file. As a result, only modification of ```false_cjtxs.json``` is required without change of (large) base files like ```coinjoin_tx_info.json``` and can be quickly recomputed.

The detection in each iteration utilizes already known false positives loaded from ```false_cjtxs.json``` file. You may download pre-prepared files for different coinjoin protocols already manually filtered by us here (file commit date corresponds approximately to ):
  - Wasabi 1.x: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/wasabi1/false_cjtxs.json)  (last coinjoin 2024-05-30)
  - Wasabi 2.x: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/wasabi2/false_cjtxs.json)  (new coinjoins still created, needs update)
  - Whirlpool: [false_cjtxs.json](https://github.com/crocs-muni/coinjoin-analysis/blob/main/data/whirlpool/false_cjtxs.json) (last coinjoin 2024-04-25, empty file, no false positives by Dumplings)

To perform one iteration of false positives detection (repeat until no new false positives are found):

#### 3.1. Run detection (this command utilizes already known false positives from ```false_cjtxs.json``` file):
```
parse_dumplings.py --cjtype ww2 --action detect_false_positives --target-path path_to_results
```
#### 3.2. Inspect created file ```no_remix_txs.json``` containing *potential* false positives

The detected potential false positives need to be manually analyzed one by one. If confirmed to be a real false positive, the transaction id shall be placed into ```false_cjtxs.json``` file to be excluded from later analyses.  
Here are some tips for detection of false positives:
  - 'both_reuse_0_70' tx are almost certainly false positives (too many addresses reused, default threshold is 70% of reused addresses, normal coinjoins have almost all addresses freshly generated). Put them all into false_cjtxs.json and rerun.
  - 'both_noremix' txs are transactions with no input and no output connected to other known coinjoin transactions. Very likely a false positive, but it needs to be analyzed one by one to confirm. 
  - txs left in "inputs_noremix" after all are typically the starting cjtx of some pool (no previous coinjoin was executed).
  - txs left in "outputs_noremix" are typically the last cjtx of some pool (either the pool closed and no longer produces transactions, or is the last mined cjtx(s) wrt Dumpling sync date)
  - after false positives are confirmed (e.g., at https://mempool.space), put them into false_cjtxs.json 

#### 3.3. Repeat the whole process again (=> smaller no_remix_txs.json). 
The typical stop point is when "both_noremix", "inputs_address_reuse_*", "outputs_address_reuse_*" and "both_reuse_*" are empty.
  
Once finished (no new false positives detected), copy ```false_cjtxs.json``` into other folders if multiple pools of the same coinjoin protocol exist (e.g., wasabi2, wasabi2_others, wasabi2_zksnacks)

<a id="plot-coinjoins"></a>
### 4. Analyze and plot results (```--action plot_coinjoins```)
To analyze and plot various analysis graphs from processed coinjoins, run:
```
parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path path_to_results
```
This command generates several files containing an analysis and visualization of executed coinjoins. For visualizations, both png and pdf file formats are generated - use *.pdf where necessary as not all details may be visible in larger *.png files. 

The files are named using the following convention: 
  - ```_values_``` means visualization of values of coinjoin inputs  
  - ```_nums_``` means visualization of number of coinjoin inputs  
  - ```_norm_``` means normalization of values before analysis  
  - ```_notnorm_``` means no normalization is performed before analysis  

The following files are generated:
  - ```*_remixrate_[values/nums]_[norm/notnorm].json``` contains remix rate (fraction of incoming value or number of inputs coming from previous coinjoins) for each coinjoin transaction. remix_ratios_all considers all inputs, remix_ratios_std considers only inputs with Wasabi 2.x standard denomination, and remix_ratios_nonstd only inputs with non-standard denomination.   
- ```*_cummul_[values/nums]_[norm/notnorm].pdf``` contains visualization of whole period aggregated per week.
  - ```*_input_[values/nums]_[norm/notnorm].pdf``` contains visualization of coinjoins splitted per each month.  

<a id="dumplings-examples"></a>
### 5. Example results
Vizualized liquidity changes in Wasabi 1.x, Wasabi 2.x and Whirlpool coinjoins 
![image](https://github.com/user-attachments/assets/33af36a6-8650-47dc-b92a-f5c611962b72)

Value of Wasabi 2.x coinjoin inputs during (March-August 2023): 
![image](https://github.com/user-attachments/assets/2a79e7ca-8a81-42c0-9c5e-3296132893c1)

Normalized ratio of different input types of Wasabi 2.x coinjoin inputs during (June-November 2023): 
![image](https://github.com/user-attachments/assets/3a6c69b8-b850-4176-973b-35422f6a111b)

Value of Wasabi 2.x coinjoins for post-zkSNACKS coordinators (June-December 2024): 
![image](https://github.com/user-attachments/assets/69ebb029-83f0-493c-bbb7-11b9b86fd746)

---

<a id="ecj-process"></a>
## Usage: Parse Wallet Wasabi 2.x emulations from EmuCoinJoin (```parse_cj_logs.py```)
The scenario assumes the previous execution of Wasabi 2.x and JoinMarket coinjoins (produced by containerized coordinator and clients) using [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) orchestration tool. 

<a id="run-ecj"></a>
### 1. Execute EmuCoinJoin emulator
See [EmuCoinJoin](https://github.com/crocs-muni/coinjoin-emulator) for a detailed setup and run of the tool.
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
Note that multiple experiments can be stored inside the ```/path_to_experiments/``` path. All found folders are checked for the ```/data/``` subfolder, and if found, the experiment is processed.

<a id="ecj-extract"></a>
### 2. Extract coinjoin information from original raw files (```--action collect_docker```)
To extract all executed coinjoins into a unified json format and perform analysis, run:
```
parse_cj_logs.py --action collect_docker --target-path path_to_experiments
```

The extraction process creates the following files: 
  * ```coinjoin_tx_info.json``` ... basic information about all detected coinjoins, mapping of all wallets to their coins, started rounds, etc.. Used for subsequent analysis.
  * ```wallets_coins.json``` ... information about every output created during execution, mapped to its coinjoin.
  * ```wallets_info.json``` ... information about every address controlled by a given wallet. 

<a id="ecj-rerun"></a>
### 3. Re-run analysis from already extracted coinjoins (```--action analyze_only```)
The coinjoin extraction part is time-consuming. If new analysis methods are added or updated, only the analysis part can be rerun. To execute again only analysis (extraction must be already done with files like ```coinjoin_tx_info.json``` already created), run:
```
parse_cj_logs.py --action analyze_only --target-path path_to_experiments
```

If the analysis finishes successfully, the following files are created:
  * ```coinjoin_stats.3.pdf, coinjoin_stats.3.pdf``` ... multiple graphs capturing various analysis results obtained from coinjoin data. 
  * ```coinjoin_tx_info_stats.json``` ... captures information about the participation of every wallet in a given coinjoin transaction.

<a id="ecj-examples"></a>
### 4. Example results
![image](https://github.com/user-attachments/assets/2e5406bc-b8f8-4725-8ff9-6484e805f682)

![image](https://github.com/user-attachments/assets/5325a4ae-468b-4b52-b58f-95d521c15b1c)

---
