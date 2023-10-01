# Scenarios

## Types of scenarios
1. SimplePassive - Everything is set before scenario starts. All wallets have same settings, but can have different starting funds.
1. ComplexPassive - Everything is set before scenario starts. Wallets can have different settings and can join or leave coinjoin in different rounds.
1. SimpleActive - Settings for wallets and backend can change each round.
1. ComplexActive - TBD

## How to run scenario
1. Fill file scenario.json with desired options and their values.
1. If you have not done it already, run init_regtest.py
1. Run your btc core, backend and client apps.
1. Run scenario.py script (need to be in same directory as scenario.json). If you changed values elsewhere (name of distributor wallet, you are not running on localhost,...), please look into scenario.py and change values in `ConstantsScenarios` class
1. Wait until scenario is finished
    * If the waiting time is too long, stop and start again. To create wallets, backend must run, so coinjoin round with old configuration is started (can take long time). It is displayed for how long you would have to wait to finish rounds with old configuration.
1. Run parse_cj_logs.py 


## Simple Passive scenario
Allowed options in scenario.json:
- type - must contain value "SimplePassive"
- freshWallets - true/false, indicates that new wallets should be created for the scenario
- startingFunds - list of integers, if option *freshWallets* is set to *true*, allows to better control starting state of the wallets. Coordinator will send Coins with these values to newly created wallets, that have no specific option in *walletsInfo*. If *freshWallets* is set to *false* this option will be ignored
- rounds - integer, number of rounds for this scenario
- walletsCounts - integer, number of wallets to be part of the scenario. If *freshWallets* is set to *true*, this number of wallets is created, otherwise, already existing wallets will be used, creating only the missing ones.
- walletsInfo - array of objects, each containing *walletIndex* and *walletFunds*. WalletIndex parameter represents index of the wallet, for which are the starting funds set. WalletFunds parameter is array of integers and represents starting funds for the specified wallet.
- walletsConfig - json, confiuration for wallets. Allowed options and allowed type of values can be seen in **parameters.json**
- backendConfig - json, configuration for backend. Allowed options and allowed type of values can be seen in **parameters.json**
