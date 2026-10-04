"""Verify producer-owned CoinJoin label sources declared by the emulator."""

import hashlib
import json
import os
import re


PRODUCER_LABEL_MANIFEST_SCHEMA_VERSION = '1.0'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def producer_label_manifest_path(data_path):
    return os.path.join(data_path, 'coinjoin_label_manifest.json')


def load_producer_label_manifest(data_path):
    """Return a manifest object, or None when processing a legacy run."""
    manifest_path = producer_label_manifest_path(data_path)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read producer label manifest {manifest_path}: {error}') from error

    if not isinstance(manifest, dict):
        raise ValueError(f'Producer label manifest must contain a JSON object: {manifest_path}')
    return manifest


def producer_label_manifest_engine(data_path):
    """Return the declared engine without requiring a manifest for legacy runs."""
    manifest = load_producer_label_manifest(data_path)
    return manifest.get('engine') if manifest is not None else None


def verify_producer_manifest_source(source, data_root, engine_name, source_is_allowed):
    """Return one integrity-checked source path constrained to the data directory."""
    relative_path = source.get('path') if isinstance(source, dict) else None
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(
            f'{engine_name} producer label manifest contains an invalid source path'
        )

    source_path = os.path.realpath(os.path.join(data_root, relative_path))
    try:
        source_escapes = os.path.commonpath((data_root, source_path)) != data_root
    except ValueError:
        source_escapes = True
    if source_escapes:
        raise ValueError(
            f'{engine_name} producer label source escapes the data directory: '
            f'{relative_path}'
        )
    if not source_is_allowed(relative_path, source_path):
        raise ValueError(
            f'{engine_name} producer label manifest contains an unexpected source: '
            f'{relative_path}'
        )
    if not os.path.isfile(source_path):
        raise ValueError(f'{engine_name} producer label source is missing: {relative_path}')

    expected_size = source.get('size_bytes')
    expected_sha256 = source.get('sha256')
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
    ):
        raise ValueError(
            f'{engine_name} producer label source metadata is invalid: {relative_path}'
        )
    if os.path.getsize(source_path) != expected_size:
        raise ValueError(
            f'{engine_name} producer label source size does not match manifest: '
            f'{relative_path}'
        )
    if sha256_file(source_path) != expected_sha256:
        raise ValueError(
            f'{engine_name} producer label source hash does not match manifest: '
            f'{relative_path}'
        )
    return source_path


def verify_producer_manifest_sources(
    data_path,
    expected_engine,
    source_is_allowed,
    reject_other_engine=False,
):
    """Return verified source paths and positive count, or None for legacy runs."""
    manifest = load_producer_label_manifest(data_path)
    if manifest is None:
        return None

    manifest_path = producer_label_manifest_path(data_path)
    engine_name = 'JoinMarket' if expected_engine == 'joinmarket' else expected_engine.capitalize()
    if manifest.get('schema_version') != PRODUCER_LABEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported producer label manifest schema version in {manifest_path}: '
            f'{manifest.get("schema_version")!r}'
        )

    if manifest.get('engine') != expected_engine:
        if reject_other_engine:
            raise ValueError(
                f'Producer label manifest engine is not {engine_name}: {manifest_path}'
            )
        return None

    if manifest.get('complete') is not True:
        reason = manifest.get('reason') or 'producer-label capture was incomplete'
        raise ValueError(f'{engine_name} producer label manifest is incomplete: {reason}')

    positive_count = manifest.get('positive_count')
    if isinstance(positive_count, bool) or not isinstance(positive_count, int) or positive_count < 0:
        raise ValueError(
            f'{engine_name} producer label manifest has invalid positive_count: '
            f'{positive_count!r}'
        )

    sources = manifest.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError(f'Complete {engine_name} producer label manifest has no sources')

    data_root = os.path.realpath(data_path)
    source_paths = []
    for source in sources:
        source_path = verify_producer_manifest_source(
            source,
            data_root,
            engine_name,
            source_is_allowed,
        )
        if source_path in source_paths:
            relative_path = source.get('path')
            raise ValueError(
                f'{engine_name} producer label manifest contains a duplicate source: '
                f'{relative_path}'
            )
        source_paths.append(source_path)

    return source_paths, positive_count
