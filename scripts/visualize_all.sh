#!/usr/bin/env bash

# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Generate static visualization pages
python3 -m cj_browser.export_browser_static --data-dir $TMP_DIR/Scanner/ --output-dir $TMP_DIR/

