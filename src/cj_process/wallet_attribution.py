"""Validate wallet ownership labels attached to parsed CoinJoin records."""

COORDINATOR_WALLET_STRING = 'Coordinator'
UNKNOWN_WALLET_STRING = 'UNKNOWN'
TXID_DISPLAY_LENGTH = 16


class CoordinatorAttributionError(RuntimeError):
    """Raised when coins are credited to a coordinator the protocol does not have."""


class UnattributedCoinsError(RuntimeError):
    """Raised when coins reach the analysis without a controlling wallet."""


def collect_records(coinjoins, predicate):
    """Locate every input/output record matching ``predicate``.

    :param coinjoins: ``coinjoins`` section of ``cjtx_stats``
    :param predicate: callable taking one input/output record, returning bool
    :return: list of ``(cjtxid, io, index, address)`` tuples
    """
    return [
        (cjtxid, io, index, record.get('address', ''))
        for cjtxid, cjtx in coinjoins.items()
        for io in ('inputs', 'outputs')
        for index, record in cjtx.get(io, {}).items()
        if predicate(record)
    ]


def format_records(records, limit=5):
    """Render a few located records, so a failure names actual coins."""
    shown = '\n'.join(
        f'    {cjtxid[:TXID_DISPLAY_LENGTH]} {io}[{index}] {address}'
        for cjtxid, io, index, address in records[:limit]
    )
    if len(records) > limit:
        shown += f'\n    ... and {len(records) - limit} more'
    return shown


def assert_all_coins_attributed(coinjoins):
    """Reject an experiment holding coins that belong to no known wallet.

    An emulated run controls every participating wallet, so a record without a
    ``wallet_name`` means its address list is incomplete -- typically a wallet
    whose keys were only partially recovered. ``fix_coordinator_wallet_addresses``
    hides the same gap for one address length only, so anything of another length
    (a Taproot address, say) arrives here unattributed and would be analyzed under
    the UNKNOWN label instead of its real owner.

    :raises UnattributedCoinsError: if any record lacks a wallet name
    """
    unattributed = collect_records(coinjoins, lambda record: not record.get('wallet_name'))
    if not unattributed:
        return

    raise UnattributedCoinsError(
        f'{len(unattributed)} record(s) belong to no known wallet. The wallet address lists '
        f'are incomplete, so these coins would be analyzed as {UNKNOWN_WALLET_STRING} rather '
        f'than their real owner:\n{format_records(unattributed)}')


def assert_no_coordinator_for_joinmarket(coinjoins):
    """Reject a JoinMarket experiment that blames coins on a coordinator.

    JoinMarket has no coordinator, so the coordinator label can only come from
    ``fix_coordinator_wallet_addresses`` papering over addresses that were never
    matched to a wallet. Those coins would silently collapse into a single fake
    participant and skew every per-wallet statistic.

    :raises CoordinatorAttributionError: if any record carries the coordinator label
    """
    mislabeled = collect_records(
        coinjoins, lambda record: record.get('wallet_name') == COORDINATOR_WALLET_STRING)
    if not mislabeled:
        return

    raise CoordinatorAttributionError(
        f'{len(mislabeled)} record(s) are attributed to {COORDINATOR_WALLET_STRING!r}, but '
        f'JoinMarket has no coordinator. The wallet address lists are incomplete, so these '
        f'coins are misattributed:\n{format_records(mislabeled)}')
