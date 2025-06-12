#!/bin/bash
BASE_PATH=$HOME

TMP_DIR="$BASE_PATH/btc/dumplings_temp2"

# Start processing in virtual environment
source $BASE_PATH/btc/coinjoin-analysis/myenv/bin/activate 

# Go to analysis folder with scripts
cd $BASE_PATH/btc/coinjoin-analysis

# Extract and process Dumplings results
python3 parse_dumplings.py --cjtype sw --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in whirlpool whirlpool_100k whirlpool_1M whirlpool_5M whirlpool_50M; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/whirlpool/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool_100k/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool_1M/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool_5M/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool_50M/fee_rates.json

# Run false positives detection
python3 parse_dumplings.py --cjtype sw --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Run generation of aggregated plots 
python3 parse_dumplings.py --cjtype sw --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 parse_dumplings.py --cjtype sw --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log


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
