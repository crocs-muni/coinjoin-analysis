from collections.abc import Mapping
import functools, orjson, sqlite3

class TxStore(Mapping):
    """
    Lazy, read-only view: txs[txid] -> dict (cached).
    Iterating over txs yields txids.  len(txs) is constant-time.
    """
    def __init__(self, db_path, cache_size=1_000):
        self._con   = sqlite3.connect(db_path, check_same_thread=False)
        self._fetch = functools.lru_cache(maxsize=cache_size)(self._load_one)

    def _load_one(self, txid: str) -> dict:
        row = self._con.execute(
            "SELECT data FROM txs WHERE txid = ?", (txid,)
        ).fetchone()
        if row is None:
            raise KeyError(txid)
        return orjson.loads(row[0])

    # ---- Mapping API --------------------------------------------------
    def __getitem__(self, txid):          # txs[txid]
        return self._fetch(txid)

    def __iter__(self):                   # for txid in txs
        for (txid,) in self._con.execute("SELECT txid FROM txs"):
            yield txid

    def __len__(self):                    # len(txs)
        return self._con.execute("SELECT COUNT(*) FROM txs").fetchone()[0]



import functools, sqlite3, msgpack
from collections.abc import Mapping

class TxStoreMsgPack(Mapping):
    def __init__(self, db_path, cache_size=1_000):
        self._con   = sqlite3.connect(db_path, check_same_thread=False)
        self._fetch = functools.lru_cache(maxsize=cache_size)(self._get)

    def _get(self, txid):
        row = self._con.execute("SELECT data FROM txs WHERE txid=?", (txid,)).fetchone()
        if row is None:
            raise KeyError(txid)
        return msgpack.unpackb(row[0], raw=False)

    # Mapping interface
    def __getitem__(self, k): return self._fetch(k)
    def __iter__(self):
        for (txid,) in self._con.execute("SELECT txid FROM txs"):
            yield txid
    def __len__(self):
        return self._con.execute("SELECT COUNT(*) FROM txs").fetchone()[0]