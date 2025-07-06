import math
import subprocess
import os
import shutil
import zipfile
from orjson import orjson
from cj_process.parse_dumplings import main as parse_dumplings_main
from cj_process.file_check import check_coinjoin_files

def run_parse_dumplings(cjtype, action, env_vars, target_path):
    arguments = []
    if cjtype:
        arguments.extend(["--cjtype", f"{cjtype}"])
    if action:
        arguments.extend(["--action", f"{action}"])
    if env_vars:
        arguments.extend(["--env_vars", f"{env_vars}"])
    if target_path:
        arguments.extend(["--target-path", f"{target_path}"])

    print(f"Running arguments: {arguments}")
    AS_SUBPROCESSS = False
    if AS_SUBPROCESSS:
        result = subprocess.run(
            ["python", "cj_process/parse_dumplings.py"] + arguments,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout, "\nSTDERR:", result.stderr)
        assert result.returncode == 0, f"cj_process/parse_dumplings.py {arguments} failed"
    else:
        returncode = parse_dumplings_main(arguments)
        assert returncode == 0, f"cj_process/parse_dumplings.py {arguments} failed"


def test_run_cj_process():
    interval_start_date = "2024-05-01 00:00:00.000000"
    interval_stop_date = "2024-06-21 00:00:00.000000"
    source_zip = os.path.abspath(os.path.join("tests", "fixtures", "dumplings__end_zksnacks_202405.zip"))
    extract_dir = os.path.abspath("../temp_dumplings")
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    #
    # Prepare test data from zip file
    #
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    shutil.copyfile(source_zip, target_zip)
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    #
    # Run initial processing
    #
    run_parse_dumplings("ww2", "process_dumplings", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    # ASSERT
    def assert_process_dumplings(coord, num_cjtxs, num_addresses, num_coins, num_distrib_values,
                                times_2097152, times_500000000, max_relative_order, max_rel_tx1, max_rel_tx2):
        with open(os.path.join(extract_dir, "Scanner", coord, "coinjoin_tx_info.json"), "r") as file:
            coinjoins = orjson.loads(file.read())
            if num_cjtxs:
                assert len(coinjoins['coinjoins']) == num_cjtxs, f"Expected {num_cjtxs} coinjoins, got {len(coinjoins['coinjoins'])}"

        if num_addresses:
            with open(os.path.join(extract_dir, "Scanner", coord, "coinjoin_tx_info_extended.json"), "r") as file:
                coinjoins = orjson.loads(file.read())
                if num_addresses:
                    assert len(coinjoins['wallets_info']['real_unknown']) == num_addresses, f"Expected {num_addresses} addresses, got {len(coinjoins['wallets_info']['real_unknown'])}"
                if num_coins:
                    assert len(coinjoins['wallets_coins']['real_unknown']) == num_coins, f"Expected {num_coins} wallets_coins, got {len(coinjoins['wallets_coins']['real_unknown'])}"

        with open(os.path.join(extract_dir, "Scanner", coord, f"{coord}_inputs_distribution.json"), "r") as file:
            distrib = orjson.loads(file.read())
            if num_distrib_values:
                assert len(distrib['distrib']) == num_distrib_values, f"Expected {num_distrib_values} values, got {len(distrib['distrib'])}"
            if times_2097152:
                assert distrib['distrib']['2097152'] == times_2097152, f"Value 2097152 expected {times_2097152} times, got {distrib['distrib']['2097152']}"
            if times_500000000:
                assert distrib['distrib']['500000000'] == times_500000000, f"Value 500000000 expected {times_500000000} times, got {distrib['distrib']['500000000']}"

        if max_relative_order:
            with open(os.path.join(extract_dir, "Scanner", coord, "cj_relative_order.json"), "r") as file:
                results = orjson.loads(file.read())
                if num_cjtxs:
                    assert len(results) == num_cjtxs, f"Expected {num_cjtxs} coinjoins, got {len(results)}"
                max_val = max(results.values())
                if max_relative_order:
                    assert max_val == max_relative_order, f"Expected max value of {max_relative_order}, got {max_val}"
                keys_with_max = [k for k, v in results.items() if v == max_val]
                if max_rel_tx1:
                    assert max_rel_tx1 in keys_with_max, f"Missing expected cjtx"
                if max_rel_tx2:
                    assert max_rel_tx2 in keys_with_max, f"Missing expected cjtx"

    assert_process_dumplings('wasabi2', 35, 9789, 9789,
                            425, 157, 2, 11,
                            "cb44436714aa5aefcbf97a2bd17e74ff2ebe885a5a472b763babd1cf471efdbe",
                            "78f3e283307fea84c735055eee6d076b13c76b224a2c0d6428e04a897d148248")
    assert_process_dumplings('wasabi2_zksnacks', 27, None, None,
                            344, 148, 2, None,
                            None, None)


    for coord in ["wasabi2", "wasabi2_others", "wasabi2_zksnacks"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join("data", "wasabi2", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Run false positives detection
    #
    run_parse_dumplings("ww2", "detect_false_positives", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    # ASSERT
    for coord in ["wasabi2"]:
        with open(os.path.join(extract_dir, "Scanner", coord, "no_remix_txs.json"), "r") as file:
            results = orjson.loads(file.read())
            assert len(results['inputs_noremix']) == 6, f"Expected {6} no inputs remix coinjoins, got {len(results['inputs_noremix'])}"
            assert len(results['outputs_noremix']) == 6, f"Expected {6} no outputs remix coinjoins, got {len(results['outputs_noremix'])}"
            assert len(results['both_noremix']) == 2, f"Expected {2} both no remix coinjoins, got {len(results['both_noremix'])}"
            assert len(results['specific_denoms_noremix_in']) == 5, f"Expected {5} specific denoms noinput in, got {len(results['specific_denoms_noremix_in'])}"
            assert len(results['specific_denoms_noremix_out']) == 6, f"Expected {6} specific denoms noinput out, got {len(results['specific_denoms_noremix_out'])}"
            assert len(results['specific_denoms_noremix_both']) == 2, f"Expected {2} specific denoms noinput both, got {len(results['specific_denoms_noremix_both'])}"
            assert len(results['inputs_address_reuse_0_70']) == 0, f"Expected {0} input address reuse, got {len(results['inputs_address_reuse_0_70'])}"
            assert len(results['outputs_address_reuse_0_70']) == 0, f"Expected {0} output address reuse, got {len(results['outputs_address_reuse_0_70'])}"

    #
    # Detect and split additional coordinators
    #
    for coord in ["wasabi2", "wasabi2_others", "wasabi2_zksnacks"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "txid_coord.json"), os.path.join(target_dir, "txid_coord.json"))
        shutil.copy(os.path.join("data", "wasabi2", "txid_coord_t.json"), os.path.join(target_dir, "txid_coord_t.json"))

    run_parse_dumplings("ww2", "detect_coordinators", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    run_parse_dumplings("ww2", "split_coordinators", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    # TODO: ASSERT 'txid_coord_discovered_renamed.json'

    # Add metadata for additional coordinators
    coords_all = ["wasabi2_kruw", "wasabi2_gingerwallet", "wasabi2_opencoordinator", "wasabi2_wasabicoordinator",
                  "wasabi2_coinjoin_nl", "wasabi2_wasabist", "wasabi2_dragonordnance", "wasabi2_mega", "wasabi2_btip", "wasabi2_strange_2025"]
    for coord in coords_all:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))
        shutil.copy(os.path.join("data", "wasabi2", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))

    #
    # Analyze liquidity
    #
    run_parse_dumplings("ww2", None, f"ANALYSIS_LIQUIDITY=True;interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    # ASSERT

    expected_results = {
        "wasabi2_zksnacks": {"total_fresh_inputs_value": 1178.30377412,
            "total_friends_inputs_value": 83.07451817, "total_unmoved_outputs_value": 1252.04660712,
            "total_leaving_outputs_value": 9.17601111, "total_nonstandard_leaving_outputs_value": 0.0,
            "total_fresh_inputs_without_nonstandard_outputs_value": 1178.30377412},
        "wasabi2_others": {"total_fresh_inputs_value": 2.35583906,
            "total_friends_inputs_value": 0.0,
            "total_unmoved_outputs_value": 2.35167402,
            "total_leaving_outputs_value": 0.0,
            "total_nonstandard_leaving_outputs_value": 0.0,
            "total_fresh_inputs_without_nonstandard_outputs_value": 2.35583906}}
    for coord in expected_results.keys():
        with open(os.path.join(extract_dir, "Scanner", f"liquidity_summary_{coord}.json"), "r") as file:
            results = orjson.loads(file.read())
            for key in expected_results[coord].keys():
                assert math.isclose(results[key], expected_results[coord][key]), f"Expected {expected_results[coord][key]} for {key}, got {results[key]}"

    #
    # Plot some graphs
    #
    run_parse_dumplings("ww2", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}",
                        extract_dir)
    # run_parse_dumplings("ww2", "plot_coinjoins",
    #                     f"PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}",
    #                     extract_dir)
    run_parse_dumplings("ww2", None,
                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}",
                        extract_dir)

    # ASSERT
    file_check = check_coinjoin_files(os.path.join(extract_dir, 'Scanner'))
    assert len(file_check['results']['wasabi2']['mix_base_files']['missing_files']) == 0, f"Missing files: {file_check['results']['wasabi2']['mix_base_files']['missing_files']}"
    assert len(file_check['results']['wasabi2_zksnacks']['mix_base_files']['missing_files']) == 3, f"Missing files: {file_check['results']['wasabi2_zksnacks']['mix_base_files']['missing_files']}"


