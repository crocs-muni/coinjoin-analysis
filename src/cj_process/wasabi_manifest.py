"""Wasabi-specific compatibility façade over producer-manifest verification."""

import os

from cj_process.producer_manifest import (
    PRODUCER_LABEL_MANIFEST_SCHEMA_VERSION,
    sha256_file,
    verify_producer_manifest_source,
    verify_producer_manifest_sources,
)


def verify_wasabi_manifest_source(source, data_root):
    """Return the verified absolute path of a single manifest source entry."""
    return verify_producer_manifest_source(
        source,
        data_root,
        engine_name='Wasabi',
        source_is_allowed=lambda _relative_path, source_path: (
            os.path.basename(source_path) == 'Logs.txt'
        ),
    )


def verify_wasabi_manifest_log_files(data_path, reject_other_engine=False):
    """Return verified manifest logs and producer count, or None for legacy runs."""
    return verify_producer_manifest_sources(
        data_path,
        expected_engine='wasabi',
        source_is_allowed=lambda _relative_path, source_path: (
            os.path.basename(source_path) == 'Logs.txt'
        ),
        reject_other_engine=reject_other_engine,
    )
