import regtest_control
import rpc_commands
import sys
import processes_control
import global_constants


if __name__ == "__main__":

    block_count = 0
    process_backend = None
    process_client = None

    try:
        block_count = regtest_control.get_block_count()

        ## 1. create wallet
        if block_count < 1:  
            regtest_control.create_wallet_btc_core("wallet")

        ## 2. generate 101 blocks
        if block_count < 101:
            regtest_control.mine_block_regtest(101)
    
    except Exception as e:
        print("An error occurred when asking BTC core for block count:", e)
        sys.exit(1)

    # 3. Create Distributor wallet

    ## 3.a Run backend and wait until it outputs creation of new block filters
    subprocesses_handler = processes_control.Wasabi_Processes_Handler()

    try:
        subprocesses_handler.run_backend()
        print("Backend sucesfully started and all filters created.")
    except Exception as e:
        print("An error occured during opening or reading output of the backend subprocess.", e)
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    # 3.b Run client
    try:
        subprocesses_handler.run_client()
        print("Client was sucesfully started and all filters were downloaded.")
    except Exception as e:
        print("An error occured during opening or reading of client output.", e)
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

    # 3.c Create Distributor wallet if not existing
    try:
        if global_constants.GLOBAL_CONSTANTS.version2:
            selection = rpc_commands.get_wallet_info(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name, False)
            selection = selection.json()
        else:
            selection = rpc_commands.select(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name, False)

        print("Response for selecting distributor wallet: ", selection)
        if "error" in selection and "not found" in selection["error"]["message"]:
            rpc_commands.create_wallet(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            print("Distributor wallet created.")

            if global_constants.GLOBAL_CONSTANTS.version2:
                rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            else:
                rpc_commands.confirmed_select(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            print("Distributor wallet loaded/selected.")

            # 4. Send funds to distributor wallet
            if global_constants.GLOBAL_CONSTANTS.version2:
                distiributor_address = rpc_commands.get_address("initiation", global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            else:
                distiributor_address = rpc_commands.get_address("initiation")
            print("Address of distributor: ", distiributor_address)
            regtest_control.send_to_address_btc_core(distiributor_address, 30)
            regtest_control.mine_block_regtest()

        elif "error" not in selection or ("error" in selection and "not fully loaded yet" in selection["error"]["message"]):
            print("Distributor already exists.")
            if global_constants.GLOBAL_CONSTANTS.version2:
                rpc_commands.confirmed_load(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            else:
                rpc_commands.confirmed_select(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            
            print("Coins:")
            if global_constants.GLOBAL_CONSTANTS.version2:
                rpc_commands.list_unspent(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
                funds = rpc_commands.get_amount_of_coins(global_constants.GLOBAL_CONSTANTS.distributor_wallet_name)
            else:
                rpc_commands.list_unspent()
                funds = rpc_commands.get_amount_of_coins()
            

            if funds < 3_000_000_000: # 2_972_333_728, 29.72_333_728
                print("Distributor has less then 30 BTC, sending additional funds")
                # 4. Send funds to distributor wallet
                if global_constants.GLOBAL_CONSTANTS.version2:
                    distiributor_address = rpc_commands.get_address("initiation", 
                                                                    global_constants.GLOBAL_CONSTANTS.distributor_wallet_name,
                                                                    verbose=False)
                else:
                    distiributor_address = rpc_commands.get_address("initiation", verbose=False)

                print("Address of distributor: ", distiributor_address)
                
                regtest_control.send_to_address_btc_core(distiributor_address, 45)
                regtest_control.mine_block_regtest()
            else:
                print("Wallet has already enough BTC")

        elif "error" in selection:
            print("Unexpected error ocurred, wasabi client rpc responded with: ", selection)
        
        else:
            print("Unexpected content of response for selecting the distributor wallet: ", selection)

    except Exception as e:
        print("An error occured during creation of distributor wallet or during the sending funds to it.", e)
        subprocesses_handler.clean_subprocesses()
        sys.exit(1)

     # 5. Cleanup - kill both running processes
    subprocesses_handler.clean_subprocesses()

