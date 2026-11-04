from pathlib import Path

import pytest

from cj_process.emulation_targets import (
    ACTION_REQUIREMENTS,
    ANALYSIS_INPUT,
    BLOCK_EXPORTS,
    DOCKER_DATA,
    EXPERIMENT_MARKERS,
    EmulationTargetError,
    Requirement,
    collect_experiments,
    find_emulation_experiments,
    validate_emulation_targets,
)


def test_requirement_describes_directories_files_and_globs():
    assert DOCKER_DATA.description == "<experiment>/data/"
    assert ANALYSIS_INPUT.description == "<experiment>/coinjoin_tx_info.json"
    assert BLOCK_EXPORTS.description == "<experiment>/data/btc-node/**/block_*.json"
    assert BLOCK_EXPORTS.summary == (
        "exported Bitcoin blocks under <experiment>/data/btc-node/**/block_*.json"
    )


def test_requirement_distinguishes_files_from_directories(tmp_path):
    (tmp_path / "coinjoin_tx_info.json").mkdir()

    assert not ANALYSIS_INPUT.is_satisfied_by(tmp_path)
    assert not DOCKER_DATA.is_satisfied_by(tmp_path)


def test_glob_requirement_ignores_directories_matching_the_pattern(tmp_path):
    block_exports = tmp_path / "data" / "btc-node"
    (block_exports / "block_1.json").mkdir(parents=True)

    assert not BLOCK_EXPORTS.is_satisfied_by(tmp_path)

    (block_exports / "block_2.json").write_text("{}", encoding="utf-8")

    assert BLOCK_EXPORTS.is_satisfied_by(tmp_path)


def test_every_action_requirement_is_a_known_marker_or_block_export():
    known = set(EXPERIMENT_MARKERS) | {BLOCK_EXPORTS}
    for requirements in ACTION_REQUIREMENTS.values():
        assert set(requirements) <= known


def test_find_emulation_experiments_is_sorted_and_skips_unrelated_folders(tmp_path):
    for name in ("experiment-2", "experiment-1"):
        (tmp_path / name / "data").mkdir(parents=True)
    (tmp_path / "notes").mkdir()
    (tmp_path / "readme.txt").write_text("", encoding="utf-8")

    assert find_emulation_experiments(tmp_path) == [
        tmp_path / "experiment-1",
        tmp_path / "experiment-2",
    ]


def test_collect_experiments_merges_several_target_paths(tmp_path):
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    (first / "experiment-1" / "WalletWasabi").mkdir(parents=True)
    (second / "experiment-2" / "WalletWasabi").mkdir(parents=True)

    assert collect_experiments([str(first), str(second)]) == [
        first / "experiment-1",
        second / "experiment-2",
    ]


def test_collect_experiments_rejects_a_path_without_experiments(tmp_path):
    populated = tmp_path / "run-a"
    (populated / "experiment-1" / "data").mkdir(parents=True)
    empty = tmp_path / "run-b"
    empty.mkdir()

    with pytest.raises(EmulationTargetError) as exc_info:
        collect_experiments([str(populated), str(empty)])

    assert str(empty) in str(exc_info.value)


def test_find_emulation_experiments_reports_unreadable_paths(tmp_path):
    with pytest.raises(EmulationTargetError, match="Cannot inspect"):
        find_emulation_experiments(Path(tmp_path / "missing"))


def test_validation_names_every_experiment_missing_an_artifact(tmp_path):
    complete = tmp_path / "experiment-1"
    (complete / "WalletWasabi").mkdir(parents=True)
    (complete / "coinjoin_tx_info.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "experiment-2"
    (incomplete / "WalletWasabi").mkdir(parents=True)

    with pytest.raises(EmulationTargetError) as exc_info:
        validate_emulation_targets([str(tmp_path)], "analyze_only")

    message = str(exc_info.value)
    assert str(incomplete) in message
    assert str(complete) not in message


def test_unknown_action_is_not_silently_accepted(tmp_path):
    (tmp_path / "experiment-1" / "data").mkdir(parents=True)

    with pytest.raises(KeyError):
        validate_emulation_targets([str(tmp_path)], "collect_everything")


def test_requirements_are_hashable_value_objects():
    assert Requirement("data") == DOCKER_DATA
    assert len({Requirement("data"), DOCKER_DATA}) == 1
