#!/bin/bash
BASE_PATH=$HOME
ROOT_BTC_DIR="${BTC_ROOT:-$HOME/btc}"
DUMPLINGS_ZIP_PATH="${DUMPLINGS_ZIP:-$ROOT_BTC_DIR/dumplings.zip}"


TMP_DIR="$ROOT_BTC_DIR/dumplings_temp2"


# Start processing in virtual environment
#source $ROOT_BTC_DIR/coinjoin-analysis/myenv/bin/activate 
source myenv/bin/activate

# Remove previous temporary directory
rm -rf $TMP_DIR/

# Create new temporary directory
mkdir $TMP_DIR/

# Unzip processed dumplings files
echo Unpacking $DUMPLINGS_ZIP_PATH to $TMP_DIR
unzip $DUMPLINGS_ZIP_PATH -d $TMP_DIR/

# Go to analysis folder with scripts
cd $ROOT_BTC_DIR/coinjoin-analysis

# Copy processed metadata 
cp $ROOT_BTC_DIR/coinjoin-analysis/data/wasabi2/wasabi2_wallet_predictions.json $TMP_DIR/Scanner/


# Extract and process Dumplings results
python3 parse_dumplings.py --cjtype ww2 --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $ROOT_BTC_DIR/coinjoin-analysis/data/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2_others/fee_rates.json
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2_zksnacks/fee_rates.json

# Run false positives detection
python3 parse_dumplings.py --cjtype ww2 --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run coordinators detection
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $ROOT_BTC_DIR/coinjoin-analysis/data/wasabi2/txid_coord.json $TMP_DIR/Scanner/$dir/
    cp $ROOT_BTC_DIR/coinjoin-analysis/data/wasabi2/txid_coord_t.json $TMP_DIR/Scanner/$dir/
done
python3 parse_dumplings.py --cjtype ww2 --action detect_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run split of post-zksnacks coordinators
python3 parse_dumplings.py --cjtype ww2 --action split_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log
# Copy fee rates into newly created folders (selected ones)
for dir in kruw gingerwallet opencoordinator wasabicoordinator coinjoin_nl wasabist dragonordnance mega btip strange_2025; do
    cp $TMP_DIR/Scanner/wasabi2/fee_rates.json $TMP_DIR/Scanner/wasabi2_$dir/
    cp $TMP_DIR/Scanner/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/wasabi2_$dir/
done

# Run detection of Bybit hack
python3 parse_dumplings.py --cjtype ww2 --env_vars="ANALYSIS_BYBIT_HACK=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Run generation of aggregated plots 
python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log
#python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;MIX_IDS=['wasabi2_zksnacks']" | tee parse_dumplings.py.log

# Run generation of multigraph plots
python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Analyse liquidity 
python3 parse_dumplings.py --cjtype ww2 --target-path $TMP_DIR/ --env_vars "ANALYSIS_LIQUIDITY=True" | tee parse_dumplings.py.log

# Visualize stats for all multigraphs
python3 parse_dumplings.py --cjtype ww2 --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2_zksnacks', 'wasabi2_kruw']" | tee parse_dumplings.py.log


# Another visualization graphs (older)
python3 parse_dumplings.py --cjtype ww2 --target-path $TMP_DIR/ --env_vars "VISUALIZE_ALL_COINJOINS_INTERVALS=True" | tee parse_dumplings.py.log



#
# Run check for created files
#
python3 file_check.py $TMP_DIR/Scanner/  | tee parse_dumplings.py.log




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