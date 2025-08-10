# Prepare expected environment
source activate_env.sh

# Run generation of aggregated plots 
python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log
#python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;MIX_IDS=['wasabi1_zksnacks']" | tee parse_dumplings.py.log


# Run generation of multigraph plots (time consuming)
#python3 -m cj_process.parse_dumplings --cjtype ww1 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

