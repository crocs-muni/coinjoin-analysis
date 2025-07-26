import helpers.global_constants as global_constants
import helpers.regtest_control as regtest_control
import subprocess
import time
import signal

class Wasabi_Processes_Handler():
    process_backend = None
    process_client = None


    def stop_client(self):
        if self.process_client is not None:
            time.sleep(2)
            self.process_client.kill()
    
    def stop_backend(self):
        if self.process_backend is not None:
            time.sleep(2)
            self.process_backend.kill()
        

    def clean_subprocesses(self):
        if self.process_client is not None:
            try:
                self.process_client.send_signal(signal.CTRL_C_EVENT)
                time.sleep(5)
            except KeyboardInterrupt:
                print("Interupt received, client stopped.")
                pass

        if self.process_backend is not None:
            try:
                self.process_backend.send_signal(signal.CTRL_C_EVENT)
                time.sleep(5)
            except KeyboardInterrupt:
                print("Interupt received, backend stopped.")
                pass


    def run_backend(self):
        block_count = 0
        try:
            block_count = regtest_control.get_block_count()
        except Exception as e:
            raise RuntimeError(e)


        self.process_backend = subprocess.Popen("dotnet run " + "--project " +  global_constants.GLOBAL_CONSTANTS.backend_folder_path, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                text=True)

        taproot_created = False
        segwit_created = False

        while True:
            output = self.process_backend.stdout.readline()
            if output == '' and self.process_backend.poll() is not None:
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
        
        #print("Created all filters.")
        self.process_backend.stdout = subprocess.DEVNULL
        

    def run_client(self):

        self.process_client = subprocess.Popen("dotnet run " + "--project " +   global_constants.GLOBAL_CONSTANTS.client_folder_path, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)

        while True:
            output = self.process_client.stdout.readline()
            if output == '' and self.process_client.poll() is not None:
                break

            # uncomment this two lines if you want to see backend output
            #if output:
            #    print(output)

            if "Downloaded filters for blocks from" in output:
                break
        self.process_client.stdout = subprocess.DEVNULL
        
        print("Downloaded filters for client")