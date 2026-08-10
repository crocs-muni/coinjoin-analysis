import pytest

from cj_process.parse_cj_logs import get_experiments_base_paths, parse_arguments


def test_target_path_must_exist(tmp_path, capsys):
    missing_path = tmp_path / "missing"

    with pytest.raises(SystemExit) as exc_info:
        parse_arguments(["--target-path", str(missing_path)])

    assert exc_info.value.code == 2
    assert "EmuCoinJoin output path does not exist" in capsys.readouterr().err


def test_target_path_must_contain_an_experiment(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_arguments(["--target-path", str(tmp_path)])

    error_output = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "expected" in error_output
    assert "<experiment>/data/" in error_output
    assert "<experiment>/WalletWasabi/" in error_output
    assert "<experiment>/coinjoin_tx_info.json" in error_output


def test_target_path_accepts_an_experiment_data_directory(tmp_path):
    btc_node = tmp_path / "experiment-1" / "data" / "btc-node"
    btc_node.mkdir(parents=True)
    (btc_node / "block_0.json").write_text("{}", encoding="utf-8")

    arguments = parse_arguments(["--target-path", str(tmp_path)])

    assert arguments.target_path == [str(tmp_path)]


def test_collect_docker_requires_exported_bitcoin_blocks(tmp_path, capsys):
    (tmp_path / "experiment-1" / "data").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        parse_arguments(["--action", "collect_docker", "--target-path", str(tmp_path)])

    assert exc_info.value.code == 2
    assert "collect_docker requires exported Bitcoin blocks" in capsys.readouterr().err


def test_collect_docker_accepts_exported_bitcoin_blocks_recursively(tmp_path):
    export_path = tmp_path / "experiment-1" / "data" / "btc-node" / "export"
    export_path.mkdir(parents=True)
    (export_path / "block_0.json").write_text("{}", encoding="utf-8")

    arguments = parse_arguments(
        ["--action", "collect_docker", "--target-path", str(tmp_path)]
    )

    assert arguments.target_path == [str(tmp_path)]


def test_collect_local_accepts_wallet_wasabi_experiment(tmp_path):
    (tmp_path / "experiment-1" / "WalletWasabi").mkdir(parents=True)

    arguments = parse_arguments(
        ["--action", "collect_local", "--target-path", str(tmp_path)]
    )

    assert arguments.target_path == [str(tmp_path)]


def test_analyze_only_requires_generated_coinjoin_info(tmp_path, capsys):
    (tmp_path / "experiment-1" / "data").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        parse_arguments(["--action", "analyze_only", "--target-path", str(tmp_path)])

    assert exc_info.value.code == 2
    assert "analyze_only requires <experiment>/coinjoin_tx_info.json" in capsys.readouterr().err


def test_analyze_only_does_not_require_exported_bitcoin_blocks(tmp_path):
    experiment = tmp_path / "experiment-1"
    experiment.mkdir()
    (experiment / "coinjoin_tx_info.json").write_text("{}", encoding="utf-8")

    arguments = parse_arguments(["--action", "analyze_only", "--target-path", str(tmp_path)])

    assert arguments.target_path == [str(tmp_path)]
    assert get_experiments_base_paths(str(tmp_path)) == [str(experiment)]
