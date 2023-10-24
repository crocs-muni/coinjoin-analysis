# Scenarios

## Types of scenarios
1. SimplePassive - Everything is set before scenario starts. All wallets have same settings, but can have different starting funds.
1. ComplexPassive - Everything is set before scenario starts. Wallets can have different settings and can join or leave coinjoin in different rounds.
1. SimpleActive - Settings for backend can change each round.

## How to run scenario
1. Fill file scenario.json with desired options and their values.
1. If you have not done it already, run init_regtest.py
1. Run your btc core.
1. Check if the constants are correctly set in files global_constants.py, scenario.py (here only version2 is important), regtest_control.py
1. Run scenario.py script. It should be in the same folder as other .py files and also scenario.json file.
1. Wait until scenario is finished
1. Run parse_cj_logs.py 


## Simple Passive scenario
Allowed options in scenario.json:
- type - must contain value "SimplePassive"
- freshWallets - true/false, indicates that new wallets should be created for the scenario
- startingFunds - list of integers, if option *freshWallets* is set to *true*, allows to better control starting state of the wallets. Coordinator will send Coins with these values to newly created wallets, that have no specific option in *walletsInfo*. If *freshWallets* is set to *false* this option will be ignored
- rounds - integer, number of rounds for this scenario (rounds that are tracked and finished, not the number of actual coinjoins)
- walletsCounts - integer, number of wallets to be part of the scenario. If *freshWallets* is set to *true*, this number of wallets is created, otherwise, already existing wallets will be used, creating only the missing ones.
- walletsInfo - array of objects, each containing *walletIndex* and *walletFunds*. WalletIndex parameter represents index of the wallet, for which are the starting funds set. WalletFunds parameter is array of integers and represents starting funds for the specified wallet.
- walletsConfig - json, confiuration for wallets. Allowed options and allowed type of values can be seen in **parameters.json**
- backendConfig - json, configuration for backend. Allowed options and allowed type of values can be seen in **parameters.json**

## Complex Passive scenario
Adds more options for walletsInfo:
- walletConfig - same possibilities as walletsConfig, apply only to specified wallet. Set before scenario is started.

Adds option of stopping and starting coinjoin in different rounds:
- TBT, needs polish

## Simple ACtive scenario
Allows changes of backend configuration:
- roundsConfigs - array of objects, each containing *index* (identifier for round) and *backendConfig* (configuration to be changed)
- need to check if all configurations can be changed during runtime or only those changing round parameters
