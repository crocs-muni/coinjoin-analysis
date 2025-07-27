#!/bin/bash
BASE_PATH=$HOME
TMP_DIR="$BASE_PATH/btc/dumplings_temp2"

#
# Extract Dumplings results
#
# Remove previous temporary directory
rm -rf $TMP_DIR/
# Create new temporary directory
mkdir $TMP_DIR/
# Unzip processed dumplings files
unzip $BASE_PATH/btc/dumplings.zip -d $TMP_DIR/



#
# Process Wasabi 2.0
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_ww2.sh

#
# Process Whirlpool Ashigaru
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_aw.sh

#
# Process JoinMarket 
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_jm.sh

#
# Process Wasabi 1.0 
#
$BASE_PATH/btc/coinjoin-analysis/scripts/process_ww1.sh




#
# Run check for created files
#
source $BASE_PATH/btc/coinjoin-analysis/venv/bin/activate 
cd $BASE_PATH/btc/coinjoin-analysis/src
python3 -m cj_process.file_check $TMP_DIR/Scanner/  | tee parse_dumplings.py.log



#
# Backup outputs
#
DEST_DIR="$BASE_PATH/data/dumplings_archive/results_$(date +%Y%m%d)"

# Get the absolute paths of source and destination
SOURCE_DIR=$(realpath "$TMP_DIR")
DEST_DIR=$(realpath "$DEST_DIR")

# Use find to locate all .json files except info_*.json and copy them while preserving structure
find "$TMP_DIR" -type f \( -name "*.json" -o -name "*.pdf" -o -name "*.png" -o -name "coinjoin_results_check_summary.txt" \) ! -name "coinjoin_tx_info*.json" ! -name "*_events.json"| while read -r file; do
    # Compute relative path
    REL_PATH="${file#$SOURCE_DIR/}"
    # Create target directory if it does not exist
    mkdir -p "$DEST_DIR/$(dirname "$REL_PATH")"
    # Copy file
    cp "$file" "$DEST_DIR/$REL_PATH"
    #echo "Copying $file to $DEST_DIR/$REL_PATH"
done
echo "Selected files archived to: $DEST_DIR"


#
# Create montage from multiple selected images
#
DEST_DIR="$BASE_PATH/data/dumplings_archive/results_$(date +%Y%m%d)"

# Wasabi2
image_list=""
for pool in others kruw gingerwallet opencoordinator coinjoin_nl wasabicoordinator wasabist dragonordnance mega btip; do
    pool_PATH="$DEST_DIR/Scanner/wasabi2_$pool/wasabi2_${pool}_cummul_values_norm.png"
    image_list="$image_list $pool_PATH"
done
montage $image_list -tile 2x -geometry +2+2 $DEST_DIR/Scanner/wasabi2/wasabi2_tiles_all_cummul_values_norm.png

# Ashigaru + JoinMarket
image_list=""
for pool in joinmarket_all whirlpool_ashigaru_2_5M whirlpool_ashigaru_25M; do
    pool_PATH="$DEST_DIR/Scanner/$pool/${pool}_cummul_values_norm.png"
    image_list="$image_list $pool_PATH"
done
montage $image_list -tile 2x -geometry +2+2 $DEST_DIR/Scanner/ashigaru_joinmarket_all_cummul_values_norm.png

#
# Upload selected files (separate scripts, can be configured based on desired upload service)
#
$BASE_PATH/btc/coinjoin-analysis/scripts/upload_results.sh


