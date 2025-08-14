import math
import os
import shutil
import utils
from pathlib import Path

from orjson import orjson

from cj_process.cj_analysis import load_coinjoins_from_file
from cj_process.file_check import check_coinjoin_files
from cj_process.file_check import check_expected_files_in_folder

TESTS = Path(__file__).resolve().parent.parent # …/repo/tests
REPO_ROOT = TESTS.parent                       # …/repo
DATA = REPO_ROOT / "data"                      # …/repo/data
TEMP_DUMPLINGS = REPO_ROOT.parent / "temp_dumplings"         # temp_dumplings
TEMP_EMUL = REPO_ROOT.parent / "temp_emul"                   # temp_emul




def assert_process_dumplings(extract_dir, coord, num_cjtxs, num_addresses, num_coins, num_distrib_values,
                             times_val1, times_val2, max_relative_order, max_rel_tx1, max_rel_tx2):
    with open(os.path.join(extract_dir, "Scanner", coord, "coinjoin_tx_info.json"), "r") as file:
        coinjoins = orjson.loads(file.read())
        if num_cjtxs:
            assert len(coinjoins[
                           'coinjoins']) == num_cjtxs, f"Expected {num_cjtxs} coinjoins, got {len(coinjoins['coinjoins'])}"

    if num_addresses:
        with open(os.path.join(extract_dir, "Scanner", coord, "coinjoin_tx_info_extended.json"), "r") as file:
            coinjoins = orjson.loads(file.read())
            if num_addresses:
                assert len(coinjoins['wallets_info'][
                               'real_unknown']) == num_addresses, f"Expected {num_addresses} addresses, got {len(coinjoins['wallets_info']['real_unknown'])}"
            if num_coins:
                assert len(coinjoins['wallets_coins'][
                               'real_unknown']) == num_coins, f"Expected {num_coins} wallets_coins, got {len(coinjoins['wallets_coins']['real_unknown'])}"

    with open(os.path.join(extract_dir, "Scanner", coord, f"{coord}_inputs_distribution.json"), "r") as file:
        distrib = orjson.loads(file.read())
        if num_distrib_values:
            assert len(distrib[
                           'distrib']) == num_distrib_values, f"Expected {num_distrib_values} values, got {len(distrib['distrib'])}"
        if times_val1:
            assert distrib['distrib'][
                       str(times_val1[0])] == times_val1[
                       1], f"Value {times_val1[0]} expected {times_val1[1]} times, got {distrib['distrib'][times_val1[0]]}"
        if times_val2:
            assert distrib['distrib'][
                       str(times_val2[0])] == times_val2[
                       1], f"Value {times_val2[0]} expected {times_val2[1]} times, got {distrib['distrib'][times_val2[0]]}"

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


def test_run_cj_process_ww2():
    interval_start_date = "2024-05-01 00:00:00.000000"
    interval_stop_date = "2024-06-21 00:00:00.000000"
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "dumplings__end_zksnacks_202405.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)


    #
    # Run initial processing
    #
    utils.run_parse_dumplings("ww2", "process_dumplings",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    assert_process_dumplings(extract_dir, 'wasabi2', 35, 9789, 9789,
                             449, (2097152, 157), (500000000, 2), 11,
                             "cb44436714aa5aefcbf97a2bd17e74ff2ebe885a5a472b763babd1cf471efdbe",
                             "78f3e283307fea84c735055eee6d076b13c76b224a2c0d6428e04a897d148248")
    assert_process_dumplings(extract_dir, 'wasabi2_zksnacks', 27, None, None,
                             367, (2097152, 148), (500000000, 2), None,
                             None, None)

    for coord in ["wasabi2", "wasabi2_others", "wasabi2_zksnacks"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi2", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi2", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Run false positives detection
    #
    utils.run_parse_dumplings("ww2", "detect_false_positives",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    for coord in ["wasabi2"]:
        with open(os.path.join(extract_dir, "Scanner", coord, "no_remix_txs.json"), "r") as file:
            results = orjson.loads(file.read())
            assert len(results[
                           'inputs_noremix']) == 6, f"Expected {6} no inputs remix coinjoins, got {len(results['inputs_noremix'])}"
            assert len(results[
                           'outputs_noremix']) == 6, f"Expected {6} no outputs remix coinjoins, got {len(results['outputs_noremix'])}"
            assert len(results[
                           'both_noremix']) == 2, f"Expected {2} both no remix coinjoins, got {len(results['both_noremix'])}"
            assert len(results[
                           'specific_denoms_noremix_in']) == 5, f"Expected {5} specific denoms noinput in, got {len(results['specific_denoms_noremix_in'])}"
            assert len(results[
                           'specific_denoms_noremix_out']) == 6, f"Expected {6} specific denoms noinput out, got {len(results['specific_denoms_noremix_out'])}"
            assert len(results[
                           'specific_denoms_noremix_both']) == 2, f"Expected {2} specific denoms noinput both, got {len(results['specific_denoms_noremix_both'])}"
            assert len(results[
                           'inputs_address_reuse_0_70']) == 0, f"Expected {0} input address reuse, got {len(results['inputs_address_reuse_0_70'])}"
            assert len(results[
                           'outputs_address_reuse_0_70']) == 0, f"Expected {0} output address reuse, got {len(results['outputs_address_reuse_0_70'])}"

    #
    # Detect and split additional coordinators
    #
    for coord in ["wasabi2", "wasabi2_others", "wasabi2_zksnacks"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi2", "txid_coord.json"), os.path.join(target_dir, "txid_coord.json"))

    utils.run_parse_dumplings("ww2", "detect_coordinators",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("ww2", "split_coordinators",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # TODO: ASSERT 'txid_coord_discovered_renamed.json'

    # Add metadata for additional coordinators
    coords_all = ["wasabi2_kruw", "wasabi2_gingerwallet", "wasabi2_opencoordinator", "wasabi2_wasabicoordinator",
                  "wasabi2_coinjoin_nl", "wasabi2_wasabist", "wasabi2_dragonordnance", "wasabi2_mega", "wasabi2_btip",
                  "wasabi2_strange_2025"]
    for coord in coords_all:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi2", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))
        shutil.copy(os.path.join(DATA, "wasabi2", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))

    #
    # Analyze liquidity
    #
    utils.run_parse_dumplings("ww2", None,
                        f"ANALYSIS_LIQUIDITY=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT

    expected_results = {
        "wasabi2_zksnacks": {"total_fresh_inputs_value": 1260.79713716,
            "total_friends_inputs_value": 0.58115513,
            "total_unmoved_outputs_value": 1252.04660712,
            "total_leaving_outputs_value": 9.17601111,
            "total_nonstandard_leaving_outputs_value": 0.0,
            "total_fresh_inputs_without_nonstandard_outputs_value": 1260.79713716},
        "wasabi2_others": {"total_fresh_inputs_value": 2.35583906,
                           "total_friends_inputs_value": 0.0,
                           "total_unmoved_outputs_value": 2.31978756,
                           "total_leaving_outputs_value": 0.03188646,
                           "total_nonstandard_leaving_outputs_value": 0.0,
                           "total_fresh_inputs_without_nonstandard_outputs_value": 2.35583906}}
    for coord in expected_results.keys():
        with open(os.path.join(extract_dir, "Scanner", f"liquidity_summary_{coord}.json"), "r") as file:
            results = orjson.loads(file.read())
            for key in expected_results[coord].keys():
                assert math.isclose(results[key], expected_results[coord][
                    key]), f"Expected {expected_results[coord][key]} for {key}, got {results[key]}"

    #
    # Plot some graphs
    #
    utils.run_parse_dumplings("ww2", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("ww2", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # utils.run_parse_dumplings("ww2", "plot_coinjoins",
    #                     f"PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
    #                     extract_dir)
    utils.run_parse_dumplings("ww2", None,
                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['wasabi2', 'wasabi2_others', 'wasabi2_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)

    # ASSERT
    file_check = check_coinjoin_files(os.path.join(extract_dir, 'Scanner'))
    assert len(file_check['results']['wasabi2']['mix_base_files'][
                   'missing_files']) == 0, f"Missing files: {file_check['results']['wasabi2']['mix_base_files']['missing_files']}"
    assert len(file_check['results']['wasabi2_zksnacks']['mix_base_files'][
                   'missing_files']) == 3, f"Missing files: {file_check['results']['wasabi2_zksnacks']['mix_base_files']['missing_files']}"


def test_run_cj_process_ww1():
    interval_start_date = "2021-08-01 00:00:00.000000"
    interval_stop_date = "2021-08-30 00:00:00.000000"

    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "dumplings__ww1_202405.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)


    #
    # Run initial processing
    #
    utils.run_parse_dumplings("ww1", "process_dumplings",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    assert_process_dumplings(extract_dir, 'wasabi1', 196, 26033, 26033,
                             4231, (333000000, 30), (500000000, 3), None,
                             None,
                             None)

    for coord in ["wasabi1"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi1", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Run false positives detection
    #
    utils.run_parse_dumplings("ww1", "detect_false_positives",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    for coord in ["wasabi1"]:
        with open(os.path.join(extract_dir, "Scanner", coord, "no_remix_txs.json"), "r") as file:
            results = orjson.loads(file.read())
            assert len(results[
                           'inputs_noremix']) == 12, f"Expected {12} no inputs remix coinjoins, got {len(results['inputs_noremix'])}"
            assert len(results[
                           'outputs_noremix']) == 6, f"Expected {6} no outputs remix coinjoins, got {len(results['outputs_noremix'])}"
            assert len(results[
                           'both_noremix']) == 2, f"Expected {2} both no remix coinjoins, got {len(results['both_noremix'])}"
            assert len(results[
                           'specific_denoms_noremix_in']) == 0, f"Expected {0} specific denoms noinput in, got {len(results['specific_denoms_noremix_in'])}"
            assert len(results[
                           'specific_denoms_noremix_out']) == 0, f"Expected {0} specific denoms noinput out, got {len(results['specific_denoms_noremix_out'])}"
            assert len(results[
                           'specific_denoms_noremix_both']) == 0, f"Expected {0} specific denoms noinput both, got {len(results['specific_denoms_noremix_both'])}"
            assert len(results[
                           'inputs_address_reuse_0_70']) == 0, f"Expected {0} input address reuse, got {len(results['inputs_address_reuse_0_70'])}"
            assert len(results[
                           'outputs_address_reuse_0_70']) == 0, f"Expected {0} output address reuse, got {len(results['outputs_address_reuse_0_70'])}"

    #
    # Detect and split additional coordinators
    #
    # utils.run_parse_dumplings("ww1", "detect_coordinators", f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'", extract_dir)
    utils.run_parse_dumplings("ww1", "split_coordinators",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # TODO: ASSERT 'txid_coord_discovered_renamed.json'

    # Remove all directories except two relevenat ones for this test
    for coord in ["wasabi1", "wasabi1_mystery", "wasabi1_others", "wasabi1_zksnacks"]:
        target_path = os.path.join(extract_dir, "Scanner", coord)
        intervals = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
        for interval in intervals:
            if (interval != "2021-08-01 00-00-00--2021-09-01 00-00-00_unknown-static-100-1utxo"
                    and interval != "2021-09-01 00-00-00--2021-10-01 00-00-00_unknown-static-100-1utxo"):
                shutil.rmtree(os.path.join(target_path, interval))

    for coord in ["wasabi1", "wasabi1_others", "wasabi1_zksnacks", "wasabi1_mystery"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "wasabi1", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))
        shutil.copy(os.path.join(DATA, "wasabi1", "txid_coord.json"), os.path.join(target_dir, "txid_coord.json"))

    #
    # Analyze liquidity
    #
    utils.run_parse_dumplings("ww1", None,
                        f"ANALYSIS_LIQUIDITY=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT

    expected_results = {
        "wasabi1_zksnacks": {"total_fresh_inputs_value": 934.54815701, "total_friends_inputs_value": 0.0,
                             "total_unmoved_outputs_value": 838.22085515, "total_leaving_outputs_value": 96.31540543,
                             "total_nonstandard_leaving_outputs_value": 39.21255878,
                             "total_fresh_inputs_without_nonstandard_outputs_value": 895.33559823},
        "wasabi1_others": {"total_fresh_inputs_value": 1800.87207463, "total_friends_inputs_value": 0.0,
                           "total_unmoved_outputs_value": 1118.58730176, "total_leaving_outputs_value": 489.54757815,
                           "total_nonstandard_leaving_outputs_value": 473.459393,
                           "total_fresh_inputs_without_nonstandard_outputs_value": 1327.41268163,
                           }}
    for coord in expected_results.keys():
        with open(os.path.join(extract_dir, "Scanner", f"liquidity_summary_{coord}.json"), "r") as file:
            results = orjson.loads(file.read())
            for key in expected_results[coord].keys():
                assert math.isclose(results[key], expected_results[coord][
                    key]), f"Expected {expected_results[coord][key]} for {key}, got {results[key]}"

    #
    # Plot some graphs
    #
    utils.run_parse_dumplings("ww1", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False;MIX_IDS=['wasabi1', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("ww1", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi1', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("ww1", None,
                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;MIX_IDS=['wasabi1', 'wasabi1_others', 'wasabi1_zksnacks'];interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)

    # ASSERT
    file_check = check_coinjoin_files(os.path.join(extract_dir, 'Scanner'))
    assert len(file_check['results']['wasabi1']['mix_base_files'][
                   'missing_files']) == 0, f"Missing files: {file_check['results']['wasabi1']['mix_base_files']['missing_files']}"
    assert len(file_check['results']['wasabi1_zksnacks']['mix_base_files'][
                   'missing_files']) == 3, f"Missing files: {file_check['results']['wasabi1_zksnacks']['mix_base_files']['missing_files']}"



def test_run_cj_process_jm():
    interval_start_date = "2024-06-01 00:00:00.000000"
    interval_stop_date = "2024-06-30 00:00:00.000000"

    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "dumplings__jm_202406.zip"))
    extract_dir = TEMP_DUMPLINGS
    target_zip = os.path.abspath(f"{extract_dir}/dumplings.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)

    #
    # Run initial processing
    #
    utils.run_parse_dumplings("jm", "process_dumplings",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    assert_process_dumplings(extract_dir, 'joinmarket_all', 51, 1676, 1676,
                             712, (546, 17), (879222944, 2), None,
                             None,
                             None)

    for coord in ["joinmarket_all"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "joinmarket", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "joinmarket", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Run false positives detection
    #
    utils.run_parse_dumplings("jm", "detect_false_positives",
                        f"interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    for coord in ["joinmarket_all"]:
        with open(os.path.join(extract_dir, "Scanner", coord, "no_remix_txs.json"), "r") as file:
            results = orjson.loads(file.read())
            assert len(results[
                           'inputs_noremix']) == 12, f"Expected {12} no inputs remix coinjoins, got {len(results['inputs_noremix'])}"
            assert len(results[
                           'outputs_noremix']) == 10, f"Expected {10} no outputs remix coinjoins, got {len(results['outputs_noremix'])}"
            assert len(results[
                           'both_noremix']) == 4, f"Expected {4} both no remix coinjoins, got {len(results['both_noremix'])}"
            assert len(results[
                           'specific_denoms_noremix_in']) == 0, f"Expected {0} specific denoms noinput in, got {len(results['specific_denoms_noremix_in'])}"
            assert len(results[
                           'specific_denoms_noremix_out']) == 0, f"Expected {0} specific denoms noinput out, got {len(results['specific_denoms_noremix_out'])}"
            assert len(results[
                           'specific_denoms_noremix_both']) == 0, f"Expected {0} specific denoms noinput both, got {len(results['specific_denoms_noremix_both'])}"
            assert len(results[
                           'inputs_address_reuse_0_70']) == 0, f"Expected {0} input address reuse, got {len(results['inputs_address_reuse_0_70'])}"
            assert len(results[
                           'outputs_address_reuse_0_70']) == 0, f"Expected {0} output address reuse, got {len(results['outputs_address_reuse_0_70'])}"


    for coord in ["joinmarket_all"]:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join(DATA, "joinmarket", "false_cjtxs.json"), os.path.join(target_dir, "false_cjtxs.json"))
        shutil.copy(os.path.join(DATA, "joinmarket", "fee_rates.json"), os.path.join(target_dir, "fee_rates.json"))

    #
    # Analyze liquidity
    #
    utils.run_parse_dumplings("jm", None,
                        f"ANALYSIS_LIQUIDITY=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    # ASSERT
    expected_results = {
        "joinmarket_all": {"total_fresh_inputs_value": 866.59799092,
            "total_friends_inputs_value": 0.0,
            "total_unmoved_outputs_value": 726.76086391,
            "total_leaving_outputs_value": 139.77221517,
            "total_nonstandard_leaving_outputs_value": 66.92181572,
            "total_fresh_inputs_without_nonstandard_outputs_value": 799.6761752}}
    for coord in expected_results.keys():
        with open(os.path.join(extract_dir, "Scanner", f"liquidity_summary_{coord}.json"), "r") as file:
            results = orjson.loads(file.read())
            for key in expected_results[coord].keys():
                assert math.isclose(results[key], expected_results[coord][
                    key]), f"Expected {expected_results[coord][key]} for {key}, got {results[key]}"

    #
    # Plot some graphs
    #
    utils.run_parse_dumplings("jm", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=False;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("jm", "plot_coinjoins",
                        f"PLOT_REMIXES_MULTIGRAPH=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)
    utils.run_parse_dumplings("jm", None,
                        f"VISUALIZE_ALL_COINJOINS_INTERVALS=True;interval_start_date='{interval_start_date}';interval_stop_date='{interval_stop_date}'",
                        extract_dir)

    # ASSERT
    file_check = check_coinjoin_files(os.path.join(extract_dir, 'Scanner'))
    assert len(file_check['results']['joinmarket_all']['mix_base_files'][
                   'missing_files']) == 2, f"Missing files: {file_check['results']['joinmarket_all']['mix_base_files']['missing_files']}"



def test_run_cj_parse_logs_ww2():
    base_experiment_name = 'grid_lognorm-static-5utxo'
    experiments = ['2024-04-05_15-19_lognorm-static-10-5utxo', '2024-04-05_17-21_lognorm-static-10-5utxo']
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "emul__ww2_lognorm-static-5utxo.zip"))
    expected_results = {
        '2024-04-05_15-19_lognorm-static-10-5utxo': {
            'len_coinjoins': 6,
            'len_wallets_info': 11,
            'num_all_coins': 260,
            'len_address_wallet_mapping': 258,
            'len_rounds': 9
        },
        '2024-04-05_17-21_lognorm-static-10-5utxo': {
            'len_coinjoins': 6,
            'len_wallets_info': 11,
            'num_all_coins': 232,
            'len_address_wallet_mapping': 231,
            'len_rounds': 8
        }}
    helper_run_cj_parse_logs(base_experiment_name, experiments, source_zip, expected_results)


def test_run_cj_parse_logs_joinmarket():
    base_experiment_name = 'joinmarket_1taker_2tumbler_40maker'
    experiments = ['2025-02-10_08-08_Joinmarket_1Taker_2Tumbler_40Maker']
    source_zip = os.path.abspath(os.path.join(TESTS, "fixtures", "emul__joinmarket_1taker_2tumbler_40maker_pruned.zip"))
    expected_results = {
        '2025-02-10_08-08_Joinmarket_1Taker_2Tumbler_40Maker': {
            'len_coinjoins': 32,
            'len_wallets_info': 44,
            'num_all_coins': 402,
            'len_address_wallet_mapping': 368,
            'len_rounds': 1
        }}
    helper_run_cj_parse_logs(base_experiment_name, experiments, source_zip, expected_results)


def helper_run_cj_parse_logs(base_experiment_name: str, experiments: list, source_zip: str, expected_results: dict):
    extract_dir = TEMP_EMUL
    target_zip = os.path.abspath(f"{extract_dir}/emul.zip")

    # Prepare test data from zip file
    utils.prepare_from_zip_file(extract_dir, source_zip, target_zip)
    #
    # Run initial processing
    #
    utils.run_parse_emul(None, "collect_docker",None, os.path.join(extract_dir, base_experiment_name))

    for experiment in experiments:
        data = load_coinjoins_from_file(os.path.join(extract_dir, base_experiment_name, experiment), None, False)
        assert len(data['coinjoins'].keys()) == expected_results[experiment]['len_coinjoins'], f"Unexpected number of coinjoins extracted: {len(data['coinjoins'].keys())}"
        assert len(data['wallets_info'].keys()) == expected_results[experiment]['len_wallets_info'], f"Unexpected number of wallets_info extracted: {len(data['wallets_info'].keys())}"
        num_all_coins = sum([len(data['wallets_coins'][wallet]) for wallet in data['wallets_coins'].keys()])
        assert num_all_coins == expected_results[experiment]['num_all_coins'], f"Unexpected number of wallets_coins extracted: {num_all_coins}"
        assert len(data['address_wallet_mapping'].keys()) == expected_results[experiment]['len_address_wallet_mapping'], f"Unexpected number of address_wallet_mapping extracted: {len(data['address_wallet_mapping'].keys())}"
        assert len(data['rounds'].keys()) == expected_results[experiment]['len_rounds'], f"Unexpected number of rounds extracted: {len(data['rounds'].keys())}"
    #
    # data = load_coinjoins_from_file(os.path.join(extract_dir, base_experiment_name, experiments[1]),  None, False)
    # assert len(data['coinjoins'].keys()) == 6, f"Unexpected number of coinjoins extracted: {len(data['coinjoins'].keys())}"
    # assert len(data['wallets_info'].keys()) == 11, f"Unexpected number of wallets_info extracted: {len(data['wallets_info'].keys())}"
    # num_all_coins = sum([len(data['wallets_coins'][wallet]) for wallet in data['wallets_coins'].keys()])
    # assert num_all_coins == 232, f"Unexpected number of wallets_coins extracted: {num_all_coins}"
    # assert len(data['address_wallet_mapping'].keys()) == 231, f"Unexpected number of address_wallet_mapping extracted: {len(data['address_wallet_mapping'].keys())}"
    # assert len(data['rounds'].keys()) == 8, f"Unexpected number of rounds extracted: {len(data['rounds'].keys())}"

    # ASSERT
    missing, _ = check_expected_files_in_folder(os.path.join(extract_dir, base_experiment_name),
                        {'aggregated_coinjoin_stats.3.pdf': None, 'aggregated_coinjoin_stats.3.png': None})
    assert len(missing) == 0, f"Missing files for {base_experiment_name}: {missing}"

    for experiment in experiments:
        missing, _ = check_expected_files_in_folder(os.path.join(extract_dir, base_experiment_name, experiment),
                            {'coinjoin_tx_info.json': None, 'wallets_coins.json': None, 'wallets_info.json': None,
                             'coinjoin_tx_info_stats.json': None, 'coinjoin_stats.3.pdf': None, 'coinjoin_stats.3.png': None})
        assert len(missing) == 0, f"Missing files for {experiment}: {missing}"

    #
    # Re-run analysis
    #

    # Remove analysis files
    os.remove(os.path.join(extract_dir, base_experiment_name, 'aggregated_coinjoin_stats.3.pdf'))
    os.remove(os.path.join(extract_dir, base_experiment_name, 'aggregated_coinjoin_stats.3.png'))
    for experiment in experiments:
        os.remove(os.path.join(extract_dir, base_experiment_name, experiment, 'coinjoin_stats.3.pdf'))
        os.remove(os.path.join(extract_dir, base_experiment_name, experiment, 'coinjoin_stats.3.png'))

    utils.run_parse_emul(None, "analyze_only",None, os.path.join(extract_dir, base_experiment_name))

    # ASSERT
    for experiment in experiments:
        missing, _ = check_expected_files_in_folder(os.path.join(extract_dir, base_experiment_name, experiment),
                            {'coinjoin_tx_info_stats.json': None, 'coinjoin_stats.3.pdf': None,
                             'coinjoin_stats.3.png': None, 'coinjoin_graph.pdf': None})
        assert len(missing) == 0, f"Missing files for {experiment}: {missing}"
