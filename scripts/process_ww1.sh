# Prepare expected environment
source activate_env.sh

# Extract and process Dumplings results
python3 -m cj_process.parse_dumplings --cjtype ww1 --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in wasabi1; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi1/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi1/fee_rates.json

# Run false positives detection
python3 -m cj_process.parse_dumplings --cjtype ww1 --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run coordinators detection - NO COORDINATION DETECTION SO FAR FOR WW1!
#for dir in wasabi1; do
#    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi1/txid_coord.json $TMP_DIR/Scanner/$dir/
#done
#python3 -m cj_process.parse_dumplings --cjtype ww1 --action detect_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Run split of coordinators
python3 -m cj_process.parse_dumplings --cjtype ww1 --action split_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy fee rates into newly created folders (selected ones)
for dir in zksnacks mystery others; do
    cp $TMP_DIR/Scanner/wasabi1/fee_rates.json $TMP_DIR/Scanner/wasabi1_$dir/
    cp $TMP_DIR/Scanner/wasabi1/false_cjtxs.json $TMP_DIR/Scanner/wasabi1_$dir/
done


# Analyse liquidity 
python3 -m cj_process.parse_dumplings --cjtype ww1 --target-path $TMP_DIR/ --env_vars "ANALYSIS_LIQUIDITY=True" | tee parse_dumplings.py.log


# Run detection of Bybit hack
python3 -m cj_process.parse_dumplings --cjtype ww1 --env_vars="ANALYSIS_BYBIT_HACK=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

