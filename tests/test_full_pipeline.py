import subprocess
import os
import tempfile
import shutil
from pathlib import Path
import zipfile

def test_run_ct_process():
    source_zip = os.path.join("tests", "fixtures", "dumplings__end_zksnacks_202505.zip")
    target_zip = "dumplings.zip"
    extract_dir = "temp_dumplings"

    # Step 1: Copy the file
    shutil.copyfile(source_zip, target_zip)

    # Step 2: Unzip into temp_dumplings
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Run the test.py script with the copied zip path
    result = subprocess.run(
        ["python",
         "cj_process/parse_dumplings.py",
         "--cjtype", "ww2",
         "--action", "process_dumplings",
         "--target-path", os.path.abspath(extract_dir)],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check that the script executed successfully
    assert result.returncode == 0, "ct_process/test.py failed"

#
# def test_dumplings_pipeline():
#     # Set up BTC_ROOT path inside CI
#     btc_root = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / "test_root"
#     os.environ["BTC_ROOT"] = str(btc_root)
#     btc_root.mkdir(parents=True, exist_ok=True)
#
#     repo_root = Path(__file__).resolve().parent.parent.parent  # Adjust if needed
#
#     # Copy the fixture zip into BTC_ROOT
#     # fixture_src = repo_root / "coinjoin-analysis" / "tests" / "fixtures" / "dumplings__end_zksnacks_202505.zip"
#     # fixture_dst = btc_root / "dumplings.zip"
#     # shutil.copy2(fixture_src, fixture_dst)
#
#     BASE_PATH = Path.home() / "btc"
#     TMP_DIR = BASE_PATH / "dumplings_temp2"
#     SCRIPT_PATH = BASE_PATH / "btc" / "cj_process" / "parse_dumplings.py"
#     SHELL_SCRIPT = repo_root / "coinjoin-analysis" / "process_test.sh"  # adjust to match real path
#
#     # Clean temp directory before test (if needed)
#     if TMP_DIR.exists():
#         shutil.rmtree(TMP_DIR)
#
#     env = os.environ.copy()
#     env["DUMPLINGS_ZIP"] = str(Path("./tests/fixtures/dumplings__end_zksnacks_202505.zip"))
#
#     # Run the entire shell script
#     result = subprocess.run(
#         ["bash", str(SHELL_SCRIPT)],
#         env=env,
#         capture_output=True,
#         text=True
#     )
#
#     # Assert script ran successfully
#     assert result.returncode == 0, f"Script failed:\n{result.stderr}"
#
#     # Assert expected output files were created
#     assert (TMP_DIR / "Scanner/wasabi2/fee_rates.json").exists(), "Missing fee_rates.json"
#     assert (TMP_DIR / "Scanner/wasabi2/false_cjtxs.json").exists(), "Missing false_cjtxs.json"
#     assert (TMP_DIR / "Scanner/wasabi2/").is_dir(), "Missing wasabi2 output dir"
#
#     # Example: check a known output plot exists
#     plots = list((TMP_DIR / "Scanner/wasabi2/plots").glob("*.pdf"))
#     assert plots, "No plots were generated"
#
#     # Optional: check log for specific messages
#     assert "Starting processing" in result.stdout or result.stderr  # adjust if needed