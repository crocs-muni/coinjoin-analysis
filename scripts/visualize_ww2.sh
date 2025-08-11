# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Run generation of aggregated plots for all coordinators
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots for single intervals 
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of (time-consuming) multigraph plots (only for selected coordinators)
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2_zksnacks', 'wasabi2_kruw']" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of multigraph plots for all coordinators (very time consuming)
#python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Another visualization graphs (older)
python3 -m cj_process.parse_dumplings --cjtype ww2 --target-path $TMP_DIR/ --env_vars "VISUALIZE_ALL_COINJOINS_INTERVALS=True" | tee parse_dumplings.py.log



