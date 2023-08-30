# Scenarios

## Types of scenarios
1. SimplePassive - Everything is set before scenario starts. All wallets have same settings.
1. ComplexPassive - Everything is set before scenario starts. Wallets can have different settings and can join or leave coinjoin in different rounds.
1. SimpleActive - Settings for wallets and backend can change each round.
1. ComplexActive - TBD

## How to run scenario
1. Fill file scenario.json with desired options and their values.
1. Run your btc core, backend and client apps.
1. Run scenario.py script (need to be in same directory as scenario.json). If you changed values elsewhere (name of distributor wallet, you are not running on localhost,...), please look into scenario.py and change values in `ConstantsScenarios` class
1. Wait until scenario is finished


## Simple Passive scenario
Allowed options in scenario.json:
- type - must contain value "SimplePassive"
- freshWallets - true/false, indicates that new wallets should be created for the scenario
- startingFunds - list of integers, if option *freshWallets* is set to *true*, allows to better control starting state of the wallets. Coordinator will send Coins with these values to all newly created wallets. If *freshWallets* is set to *false* this option will be ignored
- rounds - integer, number of rounds for this scenario
- walletsCounts - integer, number of wallets to be part of the scenario. If *freshWallets* is set to *true*, this number of wallets is created, otherwise, already existing wallets will be used, creating only the missing ones.
- backendConfig - json, configuration for backend. Allowed options and allowed type of values can be seen in **parameters.json**
- walletsConfig - json, confiuration for wallets. Allowed options and allowed type of values can be seen in **parameters.json**