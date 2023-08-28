import regtest_control
import rpc_commands
import subprocess
import sys


class InitializationConstants():
    url = "http://127.0.0.1:37128/"
    backend_folder_path = ""
    client_folder_path = ""
    distributor_wallet_name = "DistributorWallet"


INIT_REGTEST_CONSTANTS = InitializationConstants()


def clean_subprocesses(backend : subprocess.Popen, client : subprocess.Popen):
    if client is not None:
        client.kill()
    if backend is not None:
        backend.kill()
    

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
    try:
        process_backend = subprocess.Popen("dotnet run " + "--project " +  INIT_REGTEST_CONSTANTS.backend_folder_path, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                text=True)
        
        print(type(process_backend))

        taproot_created = False
        segwit_created = False

        while True:
            output = process_backend.stdout.readline()
            if output == '' and process_backend.poll() is not None:
                break

            # uncoment this two lines if you want to see backend output
            #if output:
            #    print(output)

            if f"Created Taproot filter for block: {block_count}" in output:
                taproot_created = True

            if f"Created SegwitTaproot filter for block: {block_count}" in output:
                segwit_created = True
            
            if segwit_created and taproot_created:
                break
        
        print("Created all filters.")
    
    except Exception as e:
        print("An error occured during opening or reading output of the backend subprocess.", e)
        clean_subprocesses(process_backend, process_client)
        sys.exit(1)

    # 3.b Run client
    try:
        process_client = subprocess.Popen("dotnet run " + "--project " +  INIT_REGTEST_CONSTANTS.client_folder_path, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)

        while True:
            output = process_client.stdout.readline()
            if output == '' and process_client.poll() is not None:
                break

            # uncomment this two lines if you want to see backend output
            #if output:
            #    print(output)

            if "Downloaded filters for blocks from" in output:
                break
        
        print("Downloaded filters for client")

    except:
        print("An error occured during opening or reading of client output.", e)
        clean_subprocesses(process_backend, process_client)
        sys.exit(1)

    # 3.c Create Distributor wallet if not existing
    try:
        selection = rpc_commands.select(INIT_REGTEST_CONSTANTS.distributor_wallet_name, False)
        print("Response for selecting distributor wallet: ", selection)
        if "error" in selection and "not found" in selection["error"]["message"]:
            rpc_commands.create_wallet(INIT_REGTEST_CONSTANTS.distributor_wallet_name)
            print("Distributor wallet created.")

            rpc_commands.confirmed_select(INIT_REGTEST_CONSTANTS.distributor_wallet_name)
            print("Distributor wallet selected.")

            # 4. Send funds to distributor wallet
            distiributor_address = rpc_commands.get_address("initiation")
            print("Address of distributor: ", distiributor_address)
            regtest_control.send_to_address_btc_core(distiributor_address, 30)
            regtest_control.mine_block_regtest()

        elif "error" in selection:
            print("Unexceted error ocurred, wasabi client rpc responded with: ", selection)

        else:
            print("Distributor already exists.")
            rpc_commands.confirmed_select(INIT_REGTEST_CONSTANTS.distributor_wallet_name)
            
            print("Coins:")
            rpc_commands.list_unspent()

            if rpc_commands.get_amount_of_coins() < 30:
                print("Distributor has less then 30 BTC, sending additional funds")
                # 4. Send funds to distributor wallet
                distiributor_address = rpc_commands.get_address("initiation", False)
                print("Address of distributor: ", distiributor_address)
                
                regtest_control.send_to_address_btc_core(distiributor_address, 45)
                regtest_control.mine_block_regtest()
            else:
                print("Wallet has already enough BTC")

    except Exception as e:
        print("An error occured during creation of distributor wallet or during the sending funds to it.", e)
        clean_subprocesses(process_backend, process_client)
        sys.exit(1)

     # 5. Cleanup - kill both running processes
    clean_subprocesses(process_backend, process_client)


