#!/usr/bin/env bash


# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Update mapping of transactions to coordinators from wabisator
python3 -m helpers.update_txids_with_wabisator


echo -e "\n###############################################" >> $BASE_PATH/btc/summary.log

#
# Extract Dumplings results
#
# Remove previous temporary directory
rm -rf $TMP_DIR/Scanner
# Create new temporary directory
mkdir $TMP_DIR/Scanner
# Unzip processed dumplings files
#unzip $BASE_PATH/btc/dumplings.zip -d $TMP_DIR/
unzip $BASE_PATH/dumplings.zip -d $TMP_DIR/

#
# Process Wasabi 2.0
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_ww2.sh

#
# Process Whirlpool Ashigaru
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_aw.sh

#
# Process Wasabi 1.0 
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_ww1.sh

#
# Process Samourai Whirlpool 
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_sw.sh


#
# Process JoinMarket 
# Note: Needs to come after Wasabi 1.0 and Wasabi 2.0 for false positives restoration 
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_jm.sh



#
# Visualize processed coinjoins
#
$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_ww2.sh
$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_aw.sh
$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_jm.sh
$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_ww1.sh
$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_sw.sh

$BASE_PATH/btc/coinjoin-analysis/scripts/visualize_all.sh

echo -e "\n###############################################" >> $BASE_PATH/btc/summary.log

#
# Backup and montage
#
$BASE_PATH/btc/coinjoin-analysis/scripts/backup_and_montage.sh

#
# Upload selected files (separate scripts, can be configured based on desired upload service)
#
$BASE_PATH/btc/coinjoin-analysis/scripts/upload_results.sh


echo -e "\n***********************************************" >> $BASE_PATH/btc/summary.log

