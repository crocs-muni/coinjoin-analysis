import subprocess
import os
import tempfile
import shutil
from pathlib import Path

def test_dumplings_pipeline():
    # Set up BTC_ROOT path inside CI
    btc_root = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / "test_root"
    os.environ["BTC_ROOT"] = str(btc_root)
    btc_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent  # Adjust if needed

    # Copy the fixture zip into BTC_ROOT
    # fixture_src = repo_root / "coinjoin-analysis" / "tests" / "fixtures" / "dumplings__end_zksnacks_202505.zip"
    # fixture_dst = btc_root / "dumplings.zip"
    # shutil.copy2(fixture_src, fixture_dst)


    BASE_PATH = Path.home() / "btc"
    TMP_DIR = BASE_PATH / "dumplings_temp2"
    SCRIPT_PATH = BASE_PATH / "btc" / "cj_process" / "parse_dumplings.py"
    SHELL_SCRIPT = repo_root / "coinjoin-analysis" / "process_ww2.sh"  # adjust to match real path

    # Clean temp directory before test (if needed)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    env = os.environ.copy()
    env["DUMPLINGS_ZIP"] = str(Path("./tests/fixtures/dumplings__end_zksnacks_202505.zip"))

    # Run the entire shell script
    result = subprocess.run(
        ["bash", str(SHELL_SCRIPT)],
        env=env,
        capture_output=True,
        text=True
    )

    # Assert script ran successfully
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    # Assert expected output files were created
    assert (TMP_DIR / "Scanner/wasabi2/fee_rates.json").exists(), "Missing fee_rates.json"
    assert (TMP_DIR / "Scanner/wasabi2/false_cjtxs.json").exists(), "Missing false_cjtxs.json"
    assert (TMP_DIR / "Scanner/wasabi2/").is_dir(), "Missing wasabi2 output dir"

    # Example: check a known output plot exists
    plots = list((TMP_DIR / "Scanner/wasabi2/plots").glob("*.pdf"))
    assert plots, "No plots were generated"

    # Optional: check log for specific messages
    assert "Starting processing" in result.stdout or result.stderr  # adjust if needed