"""Verification of JoinMarket sources in the emulator producer-label manifest."""

import os

from cj_process.producer_manifest import verify_producer_manifest_sources


JOINMARKET_ROUND_EVENTS_SOURCE = 'joinmarket_round_events.json'


def verify_joinmarket_manifest_round_events_file(base_path):
    """Return the verified event file and positive count, or None for legacy runs."""
    verified = verify_producer_manifest_sources(
        os.path.join(base_path, 'data'),
        expected_engine='joinmarket',
        source_is_allowed=lambda relative_path, _source_path: (
            relative_path == JOINMARKET_ROUND_EVENTS_SOURCE
        ),
        reject_other_engine=True,
    )
    if verified is None:
        return None

    source_paths, positive_count = verified
    if len(source_paths) != 1:
        raise ValueError(
            'JoinMarket producer label manifest must contain exactly one '
            f'{JOINMARKET_ROUND_EVENTS_SOURCE} source'
        )
    return source_paths[0], positive_count
