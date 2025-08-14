import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# IDEA for simpler case specific for coinjoin-analysis case:
# 1. top most folder(s) == mix_name(s)
# 2. mix_name expected in file name
# 3. sub-folders of mix_name matching interval folders => interval_name
# 4. $mix_name$/$interval_name$/$mix_name$_file_name.ext checked


# Cummulative liquidity graphs
graphs_cummulative_liquidity = {
    "$MIX_NAME$_cummul_nums_norm.pdf": 10240,
    "$MIX_NAME$_cummul_nums_norm.png": 10240,
    "$MIX_NAME$_cummul_nums_notnorm.pdf": 10240,
    "$MIX_NAME$_cummul_nums_notnorm.png": 10240,
    "$MIX_NAME$_cummul_values_norm.pdf": 10240,
    "$MIX_NAME$_cummul_values_norm.png": 10240,
    "$MIX_NAME$_cummul_values_notnorm.pdf": 10240,
    "$MIX_NAME$_cummul_values_notnorm.png": 10240
}

graphs_full_liquidity = {
    "$MIX_NAME$_input_types_nums_norm.pdf": 10240,
    "$MIX_NAME$_input_types_nums_norm.png": 10240,
    "$MIX_NAME$_input_types_nums_notnorm.pdf": 10240,
    "$MIX_NAME$_input_types_nums_notnorm.png": 10240,
    "$MIX_NAME$_input_types_values_norm.pdf": 10240,
    "$MIX_NAME$_input_types_values_norm.png": 10240,
    "$MIX_NAME$_input_types_values_notnorm.pdf": 10240,
    "$MIX_NAME$_input_types_values_notnorm.png": 10240
}

distrib_fresh_liquidity = {
    "$MIX_NAME$_freshliquidity_nums_norm.json": 10240,
    "$MIX_NAME$_freshliquidity_nums_notnorm.json": 10240,
    "$MIX_NAME$_freshliquidity_values_norm.json": 10240,
    "$MIX_NAME$_freshliquidity_values_notnorm.json": 10240
}

distrib_remix_rate = {
    "$MIX_NAME$_remixrate_nums_norm.json": 10240,
    "$MIX_NAME$_remixrate_nums_notnorm.json": 10240,
    "$MIX_NAME$_remixrate_values_norm.json": 10240,
    "$MIX_NAME$_remixrate_values_notnorm.json": 10240
}

expected_files = {
    "mix_base_files": {
        "coinjoin_tx_info.json": 10240,
        "coinjoin_tx_info_extended.json": 10240,
        "false_cjtxs.json": 10240,
        "fee_rates.json": 10240,
        "cj_relative_order.json": 10240,
        "no_remix_txs.json": 10240,
        "no_remix_txs_simplified.json": 10240,
        "txid_coord.json": 10240,

        "$MIX_NAME$_coinjoin_stats.pdf": 10240,
        "$MIX_NAME$_coinjoin_stats.png": 10240,

        "$MIX_NAME$_inputs_distribution.json": 10240,   # Distribution of fresh mix input values

        **graphs_full_liquidity,

        **distrib_fresh_liquidity,
        **distrib_remix_rate,
    },
    "mix_interval_files": {     # Files expected in single interval folder
        "coinjoin_tx_info.json": 10240,
        "$MIX_NAME$_events.json": 10240,
        **graphs_full_liquidity,
    }
}


def replace_name_in_dict(data, str_to_replace, str_value):
    """
    Recursively replace all occurrences of $name$ in keys and values of a dictionary.

    Parameters:
        data (dict): The dictionary to process.
        str_value (str): The value to replace $name$ with.

    Returns:
        dict: A new dictionary with replacements made.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = key.replace(str_to_replace, str_value) if isinstance(key, str) else key
            new_value = replace_name_in_dict(value, str_to_replace, str_value)
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(data, list):
        return [replace_name_in_dict(item, str_to_replace, str_value) for item in data]
    elif isinstance(data, str):
        return data.replace(str_to_replace, str_value)
    else:
        return data


def check_expected_files_in_folder(folder_path, expected_files):
    # Load files in folder
    files = [f for f in Path(folder_path).iterdir() if f.is_file()]
    files_names = {f.name: f for f in files}
    # Check against expected list
    missing_files = [file for file in expected_files.keys() if file not in files_names.keys()]
    found_files = [file for file in expected_files.keys() if file in files_names.keys()]

    return missing_files, found_files


def check_coinjoin_files(base_path):
    print(f'Processing check_coinjoin_files({base_path})')
    mix_results_check = {}
    all_status = ''
    # Get top-level folders => mix_names
    all_mixes = [f for f in Path(base_path).iterdir() if f.is_dir()]

    # For each mix, perform expansion of required files and check
    for mix_path in all_mixes:
        mix_status = mix_path.name + ' : '
        mix_results_check[mix_path.name] = {'mix_base_files': {'missing_files': [], 'found_files': {}},
                                            'mix_interval_files': {'missing_files': [], 'found_files': {}}}

        # Prepare expected files list customized for mix_id
        expected_files_mix = copy.deepcopy(expected_files)
        expected_files_mix = replace_name_in_dict(expected_files_mix, '$MIX_NAME$', mix_path.name)

        #
        # Check mix_base_files
        #
        mix_base_path = os.path.join(base_path, mix_path.name)
        missing_files, found_files = check_expected_files_in_folder(mix_base_path, expected_files_mix['mix_base_files'])

        mix_results_check[mix_path.name]['mix_base_files']['missing_files'].extend([Path(f).as_posix() for f in missing_files])
        mix_results_check[mix_path.name]['mix_base_files']['found_files'].update({Path(f).as_posix(): Path(os.path.join(mix_base_path, f)).stat().st_size for f in found_files})
        mix_results_check[mix_path.name]['mix_base_files']['score'] = f"{round(100 * len(found_files)/len(expected_files_mix['mix_base_files']), 2)}"
        mix_results_check[mix_path.name]['mix_base_files']['score_str'] = f"{len(found_files)}/{len(expected_files_mix['mix_base_files'])}"

        mix_status += mix_results_check[mix_path.name]['mix_base_files']['score_str']
        mix_status += '' if len(found_files) == len(expected_files_mix['mix_base_files']) else '[!]'
        mix_status += '__'
        #
        # Check intervals files
        #
        # Get all subfolders, if interval pattern, then check files inside
        interval_found_num = 0
        interval_expected_num = 0
        all_folders = [f for f in Path(mix_path).iterdir() if f.is_dir()]
        for interval_folder in all_folders:
            if '_unknown-static-100-1utxo' in interval_folder.name:  # detect interval
                missing_files, found_files = check_expected_files_in_folder(interval_folder,
                                                                            expected_files_mix['mix_interval_files'])
                mix_results_check[mix_path.name]['mix_interval_files']['missing_files'].extend(
                    [Path(os.path.join(interval_folder.name, f)).as_posix() for f in missing_files])
                mix_results_check[mix_path.name]['mix_interval_files']['found_files'].update(
                    {Path(os.path.join(interval_folder, f)).as_posix(): Path(os.path.join(interval_folder, f)).stat().st_size for f in found_files})
                interval_found_num += len(found_files)
                interval_expected_num += len(expected_files_mix['mix_interval_files'])

        mix_results_check[mix_path.name]['mix_interval_files']['score'] = f'{round(100 * interval_found_num/interval_expected_num, 2)}' if interval_expected_num > 0 else '0'
        mix_results_check[mix_path.name]['mix_interval_files']['score_str'] = f'{interval_found_num}/{interval_expected_num}'
        mix_status += mix_results_check[mix_path.name]['mix_interval_files']['score_str']
        mix_status += '' if interval_found_num == interval_expected_num else '[!]'
        mix_status += '   '

        all_status += '[CHECK] ' if '[!]' in mix_status else '[OK] '
        all_status += mix_status

    mix_results_all = {'results': mix_results_check}
    mix_results_all['info'] = {'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'base_path': base_path, 'status': all_status}
    with open(os.path.join(base_path, "coinjoin_results_check.json"), "w") as file:
        file.write(json.dumps(dict(sorted(mix_results_all.items())), indent=4))

    all_status = all_status.replace('   ', '\n')
    print(f'{all_status}')
    with open(os.path.join(base_path, "coinjoin_results_check_summary.txt"), "w") as file:
        file.write(all_status)

    return mix_results_all


if __name__ == "__main__":
    check_coinjoin_files(sys.argv[1])

