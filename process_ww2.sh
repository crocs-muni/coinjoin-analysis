#!/bin/bash

TMP_DIR="$HOME/btc/dumplings_temp"

# Remove previous temporary directory
rm -rf $TMP_DIR/

# Create new temporary directory
mkdir $TMP_DIR/

# Unzip processed dumplings files
unzip $HOME/btc/dumplings.zip -d $TMP_DIR/

# Start processing in virtual environment
source $HOME/btc/coinjoin-analysis/myenv/bin/activate 

# Go to analysis folder with scripts
cd $HOME/btc/coinjoin-analysis

# Copy processed metadata 
cp $HOME/btc/coinjoin-analysis/data/wasabi2/wasabi2_wallet_predictions.json $TMP_DIR/Scanner/


# Extract and process Dumplings results
python3 parse_dumplings.py --cjtype ww2 --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $HOME/btc/coinjoin-analysis/data/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2_others/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2_zksnacks/fee_rates.json

# Run false positives detection
python3 parse_dumplings.py --cjtype ww2 --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run coordinators detection
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $HOME/btc/coinjoin-analysis/data/wasabi2/txid_coord.json $TMP_DIR/Scanner/$dir/
done
python3 parse_dumplings.py --cjtype ww2 --action detect_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run split of post-zksnacks coordinators
python3 parse_dumplings.py --cjtype ww2 --action split_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log
# Copy fee rates into newly created folders (selected ones)
for dir in kruw gingerwallet opencoordinator wasabicoordinator coinjoin_nl wasabist dragonordnance mega btip; do
    cp $TMP_DIR/Scanner/wasabi2/fee_rates.json $TMP_DIR/Scanner/wasabi2_$dir/
    cp $TMP_DIR/Scanner/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/wasabi2_$dir/
done

# Run generation of plots
python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path $TMP_DIR/ | tee parse_dumplings.py.log


#
# Backup outputs
#
DEST_DIR="/data/btc/dumplings_archive/results_$(date +%Y%m%d)"
echo "(Re-)creating $DEST_DIR"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# Get the absolute paths of source and destination
SOURCE_DIR=$(realpath "$TMP_DIR")
DEST_DIR=$(realpath "$DEST_DIR")

# Use find to locate all .json files except info_*.json and copy them while preserving structure
find "$TMP_DIR" -type f \( -name "*.json" -o -name "*.pdf" -o -name "*.png" \) ! -name "coinjoin_tx_info*.json" ! -name "*_events.json"| while read -r file; do
    # Compute relative path
    REL_PATH="${file#$SOURCE_DIR/}"
    # Create target directory if it does not exist
    mkdir -p "$DEST_DIR/$(dirname "$REL_PATH")"
    # Copy file
    cp "$file" "$DEST_DIR/$REL_PATH"
    #echo "Copying $file to $DEST_DIR/$REL_PATH"
done

echo "Selected files archived to: $DEST_DIR"

# Create montage from multiple selected images
DEST_DIR="/data/btc/dumplings_archive/results_$(date +%Y%m%d)"
image_list=""
for pool in others kruw gingerwallet opencoordinator coinjoin_nl wasabicoordinator wasabist dragonordnance mega btip; do
    pool_PATH="$DEST_DIR/Scanner/wasabi2_$pool/wasabi2_${pool}_cummul_values_norm.png"
    image_list="$image_list $pool_PATH"
done
#for pool in others kruw gingerwallet opencoordinator; do
#    pool_PATH="$DEST_DIR/Scanner/wasabi2_$pool/wasabi2_${pool}_input_types_values_norm.png"
#    image_list="$image_list $pool_PATH"
#done
montage $image_list -tile 2x -geometry +2+2 $DEST_DIR/Scanner/wasabi2/wasabi2_tiles_all_cummul_values_norm.png

# Upload selected files (separate scripts, can be configured based on desired upload service)
./upload_ww2.sh