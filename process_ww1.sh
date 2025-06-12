#!/bin/bash
BASE_PATH=$HOME

TMP_DIR="$BASE_PATH/btc/dumplings_temp2"

# Start processing in virtual environment
source $BASE_PATH/btc/coinjoin-analysis/myenv/bin/activate 

# Go to analysis folder with scripts
cd $BASE_PATH/btc/coinjoin-analysis

# Copy processed metadata 
#cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/wasabi2_wallet_predictions.json $TMP_DIR/Scanner/


# Extract and process Dumplings results
python3 parse_dumplings.py --cjtype ww1 --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in wasabi1; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi1/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi1/fee_rates.json

# Run false positives detection
python3 parse_dumplings.py --cjtype ww1 --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run coordinators detection
for dir in wasabi1; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi1/txid_coord.json $TMP_DIR/Scanner/$dir/
done
python3 parse_dumplings.py --cjtype ww1 --action detect_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Run split of coordinators
python3 parse_dumplings.py --cjtype ww1 --action split_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log
# Copy fee rates into newly created folders (selected ones)
for dir in zksnacks mystery others; do
    cp $TMP_DIR/Scanner/wasabi1/fee_rates.json $TMP_DIR/Scanner/wasabi1_$dir/
    cp $TMP_DIR/Scanner/wasabi1/false_cjtxs.json $TMP_DIR/Scanner/wasabi1_$dir/
done






# Run detection of Bybit hack
python3 parse_dumplings.py --cjtype ww1 --env_vars="ANALYSIS_BYBIT_HACK=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Run generation of aggregated plots 
python3 parse_dumplings.py --cjtype ww1 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 parse_dumplings.py --cjtype ww1 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log
#python3 parse_dumplings.py --cjtype ww1 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;MIX_IDS=['wasabi2_zksnacks']" | tee parse_dumplings.py.log


#
# Backup outputs
#
DEST_DIR="/data/btc/dumplings_archive/results_$(date +%Y%m%d)"

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
