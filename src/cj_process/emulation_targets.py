"""Discovery and validation of EmuCoinJoin experiment directories.

A ``--target-path`` points at a folder whose one-level subfolders are the
experiments. Which artifacts an experiment has to provide depends on the
requested action, so the requirements are kept as data in
``ACTION_REQUIREMENTS`` and the same markers drive experiment discovery.
"""

from dataclasses import dataclass
from pathlib import Path


class EmulationTargetError(ValueError):
    """Raised when a target path does not hold experiments usable for an action."""


@dataclass(frozen=True)
class Requirement:
    """One artifact an experiment must provide before an action can run."""

    path: str
    glob: str = ''
    must_be_file: bool = False
    label: str = ''

    @property
    def description(self) -> str:
        """Path of the artifact relative to a single experiment folder."""
        if self.glob:
            return f'<experiment>/{self.path}/**/{self.glob}'
        if self.must_be_file:
            return f'<experiment>/{self.path}'
        return f'<experiment>/{self.path}/'

    @property
    def summary(self) -> str:
        """Human readable phrase used in error messages."""
        return f'{self.label} {self.description}'.strip()

    def is_satisfied_by(self, experiment: Path) -> bool:
        target = experiment / self.path
        if self.glob:
            return any(match.is_file() for match in target.rglob(self.glob))
        return target.is_file() if self.must_be_file else target.is_dir()


DOCKER_DATA = Requirement('data')
WALLET_WASABI = Requirement('WalletWasabi')
ANALYSIS_INPUT = Requirement('coinjoin_tx_info.json', must_be_file=True)
BLOCK_EXPORTS = Requirement('data/btc-node', glob='block_*.json',
                            label='exported Bitcoin blocks under')

# Any of these makes a subfolder recognizable as an experiment.
EXPERIMENT_MARKERS = (DOCKER_DATA, WALLET_WASABI, ANALYSIS_INPUT)

DEFAULT_ACTION = 'collect_docker'
ACTION_REQUIREMENTS = {
    'collect_local': (WALLET_WASABI,),
    'collect_docker': (DOCKER_DATA, BLOCK_EXPORTS),
    'analyze_only': (ANALYSIS_INPUT,),
}


def is_experiment(path: Path) -> bool:
    """Decide whether a folder holds an experiment recognized by any workflow."""
    return path.is_dir() and any(marker.is_satisfied_by(path) for marker in EXPERIMENT_MARKERS)


def find_emulation_experiments(path: Path):
    """List experiment folders directly under a single target path."""
    try:
        return sorted(experiment for experiment in path.iterdir() if is_experiment(experiment))
    except OSError as exc:
        raise EmulationTargetError(
            f'Cannot inspect EmuCoinJoin output path {path}: {exc}'
        ) from exc


def collect_experiments(target_paths):
    """Gather experiments from every target path, requiring each one to hold some."""
    experiments = []
    for target_path in target_paths:
        path = Path(target_path)
        if not path.is_dir():
            raise EmulationTargetError(
                f'EmuCoinJoin output path does not exist: {target_path}'
            )

        found = find_emulation_experiments(path)
        if not found:
            expected = ' or '.join(f'{target_path}/{marker.description}'
                                   for marker in EXPERIMENT_MARKERS)
            raise EmulationTargetError(
                f'No EmuCoinJoin experiments found under {target_path}; expected {expected}'
            )
        experiments.extend(found)

    return experiments


def verify_requirement(experiments, action: str, requirement: Requirement):
    """Fail when any experiment lacks the artifact demanded by the action."""
    missing = [str(experiment) for experiment in experiments
               if not requirement.is_satisfied_by(experiment)]
    if missing:
        raise EmulationTargetError(
            f'{action} requires {requirement.summary}; missing for: ' + ', '.join(missing)
        )


def validate_emulation_targets(target_paths, action):
    """Validate all target paths against the requirements of the requested action."""
    effective_action = action or DEFAULT_ACTION
    experiments = collect_experiments(target_paths)
    for requirement in ACTION_REQUIREMENTS[effective_action]:
        verify_requirement(experiments, effective_action, requirement)
