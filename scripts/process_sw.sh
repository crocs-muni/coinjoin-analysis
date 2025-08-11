# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Extract and process Dumplings results
python3 -m cj_process.parse_dumplings --cjtype sw --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in whirlpool whirlpool_100k whirlpool_1M whirlpool_5M whirlpool_50M whirlpool_ashigaru_25M whirlpool_ashigaru_2_5M; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/whirlpool/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/whirlpool/fee_rates.json
for dir in whirlpool_100k whirlpool_1M whirlpool_5M whirlpool_50M whirlpool_ashigaru_25M whirlpool_ashigaru_2_5M; do
    cp $TMP_DIR/Scanner/whirlpool/fee_rates.json $TMP_DIR/Scanner/$dir/fee_rates.json
done

# Run false positives detection
python3 -m cj_process.parse_dumplings --cjtype sw --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Analyse liquidity 
python3 -m cj_process.parse_dumplings --cjtype sw --target-path $TMP_DIR/ --env_vars "ANALYSIS_LIQUIDITY=True" | tee parse_dumplings.py.log
