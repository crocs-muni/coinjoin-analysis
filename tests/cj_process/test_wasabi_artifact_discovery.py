from cj_process import parse_cj_logs


def test_finds_split_wasabi_coordinator_log_without_manifest(tmp_path):
    log_path = tmp_path / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("coordinator log\n", encoding="utf-8")

    assert parse_cj_logs.find_wasabi_coordinator_log_files(str(tmp_path)) == [
        str(log_path.resolve())
    ]


def test_finds_legacy_wasabi_backend_log_without_manifest(tmp_path):
    log_path = tmp_path / "data" / "wasabi-backend" / "backend" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("backend log\n", encoding="utf-8")

    assert parse_cj_logs.find_wasabi_coordinator_log_files(str(tmp_path)) == [
        str(log_path.resolve())
    ]
