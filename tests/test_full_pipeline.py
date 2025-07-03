import subprocess
import os
import tempfile
import shutil
from pathlib import Path
import zipfile
import requests

def run_parse_dumplings(cjtype, action, env_vars, target_path):
    arguments = ["python", "cj_process/parse_dumplings.py"]
    if cjtype:
        arguments.extend(["--cjtype", f"{cjtype}"])
    if action:
        arguments.extend(["--action", f"{action}"])
    if env_vars:
        arguments.extend(["--env_vars", f"{env_vars}"])
    if target_path:
        arguments.extend(["--target-path", f"{target_path}"])

    print(f"Running arguments: {arguments}")
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout, "\nSTDERR:", result.stderr)
    assert result.returncode == 0, f"cj_process/parse_dumplings.py {arguments} failed"


def test_run_cj_process():
    interval_start_date = "2024-05-01 00:00:00.000000"
    interval_stop_date = "2024-06-21 00:00:00.000000"
    source_zip = os.path.abspath(os.path.join("tests", "fixtures", "dumplings__end_zksnacks_202505.zip"))
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

    coords = ["wasabi2", "wasabi2_others", "wasabi2_zksnacks"]

    # Copy false_cjtxs.json into each directory
    for coord in coords:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "false_cjtxs.json"), os.path.join(extract_dir, "Scanner", coord, "false_cjtxs.json"))
        shutil.copy(os.path.join("data", "wasabi2", "fee_rates.json"), os.path.join(extract_dir, "Scanner", coord, "fee_rates.json"))

    #
    # Run false positives detection
    #
    run_parse_dumplings("ww2", "detect_false_positives", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)

    # Copy known coordinators files
    for coord in coords:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "txid_coord.json"), os.path.join(extract_dir, "Scanner", coord, "txid_coord.json"))
        shutil.copy(os.path.join("data", "wasabi2", "txid_coord_t.json"), os.path.join(extract_dir, "Scanner", coord, "txid_coord_t.json"))

    #
    # Detect and split additional coordinators
    #
    run_parse_dumplings("ww2", "detect_coordinators", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    run_parse_dumplings("ww2", "split_coordinators", f"interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)

    # Add metadata for additional coordinators
    coords_all = ["wasabi2_kruw", "wasabi2_gingerwallet", "wasabi2_opencoordinator", "wasabi2_wasabicoordinator",
                  "wasabi2_coinjoin_nl", "wasabi2_wasabist", "wasabi2_dragonordnance", "wasabi2_mega", "wasabi2_btip", "wasabi2_strange_2025"]
    for coord in coords_all:
        target_dir = os.path.join(extract_dir, "Scanner", coord)
        shutil.copy(os.path.join("data", "wasabi2", "fee_rates.json"), os.path.join(extract_dir, "Scanner", coord, "fee_rates.json"))
        shutil.copy(os.path.join("data", "wasabi2", "false_cjtxs.json"), os.path.join(extract_dir, "Scanner", coord, "false_cjtxs.json"))

    #
    # Plot some graphs
    #
    run_parse_dumplings("ww2", "plot_coinjoins", f"PLOT_REMIXES_MULTIGRAPH=False;interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)
    #run_parse_dumplings("ww2", "plot_coinjoins", f"PLOT_REMIXES_SINGLE_INTERVAL=True;interval_stop_date={interval_stop_date}", extract_dir)


    run_parse_dumplings("ww2", None, f"ANALYSIS_LIQUIDITY=True;interval_start_date={interval_start_date};interval_stop_date={interval_stop_date}", extract_dir)

    #
    # Verify expected results
    #
    for coord in coords:
        # Assert expected output files were created
        assert (extract_dir / f"Scanner/{coord}/fee_rates.json").exists(), f"Missing {coord}/fee_rates.json"
        assert (extract_dir / f"Scanner/{coord}/false_cjtxs.json").exists(), f"Missing {coord} false_cjtxs.json"
        assert (extract_dir / f"Scanner/{coord}/").is_dir(), f"Missing {coord} output dir"

        # Example: check a known output plots exist
        plots_pdf = list((f"{extract_dir}/Scanner/{coord}").glob("*.pdf"))
        assert plots_pdf, "No pdf plots were generated"
        plots_png = list((f"{extract_dir}/Scanner/{coord}").glob("*.png"))
        assert plots_png, "No png plots were generated"
