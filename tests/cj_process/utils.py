import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from cj_process.parse_dumplings import main as parse_dumplings_main
from cj_process.parse_cj_logs import main as parse_cj_logs_main


def write_manifest(run_dir, log_path, positive_count=0, complete=True):
    data_dir = run_dir / "data"
    manifest = {
        "schema_version": "1.0",
        "engine": "wasabi",
        "complete": complete,
        "reason": None if complete else "coordinator log capture failed",
        "positive_count": positive_count,
        "sources": [{
            "path": log_path.relative_to(data_dir).as_posix(),
            "size_bytes": log_path.stat().st_size,
            "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }],
    }
    (data_dir / "coinjoin_label_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def prepare_from_zip_file(extract_dir: Path, source_zip, target_zip):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    shutil.copyfile(source_zip, target_zip)
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def run_parse_dumplings(cjtype, action, env_vars, target_path, must_succeed=True):
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
    AS_SUBPROCESSS = False  # If True, Main is executed as subprocess
    if AS_SUBPROCESSS:
        result = subprocess.run(
            ["python", "cj_process/parse_dumplings.py"] + arguments,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout, "\nSTDERR:", result.stderr)
        if must_succeed:
            assert result.returncode == 0, f"cj_process/parse_dumplings.py {arguments} failed"
        return result.returncode
    else:
        # Call directly within this context
        returncode = parse_dumplings_main(arguments)
        if must_succeed:
            assert returncode == 0, f"cj_process/parse_dumplings.py {arguments} failed"
        return returncode



def run_parse_emul(cjtype, action, env_vars, target_path, must_succeed=True):
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
    AS_SUBPROCESSS = False  # If True, Main is executed as subprocess
    if AS_SUBPROCESSS:
        result = subprocess.run(
            ["python", "c:/!blockchains/CoinJoin/coinjoin-analysis/src/cj_process/parse_cj_logs.py"] + arguments,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout, "\nSTDERR:", result.stderr)
        if must_succeed:
            assert result.returncode == 0, f"cj_process/parse_cj_logs.py {arguments} failed"
        return result.returncode
    else:
        # Call directly within this context
        returncode = parse_cj_logs_main(arguments)
        if must_succeed:
            assert returncode == 0, f"cj_process/parse_cj_logs.py {arguments} failed"
        return returncode
