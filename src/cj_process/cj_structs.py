import logging
from datetime import datetime
from enum import Enum, IntFlag, auto

class MIX_EVENT_TYPE(Enum):
    MIX_ENTER = 'MIX_ENTER'  # New liquidity coming to mix
    MIX_LEAVE = 'MIX_LEAVE'  # Liquidity leaving mix (postmix spend)
    MIX_REMIX = 'MIX_REMIX'  # Remixed value within mix
    MIX_REMIX_FRIENDS = 'MIX_REMIX_FRIENDS'  # Remixed value within mix, but not directly, but one hop friends (WW2)
    MIX_REMIX_FRIENDS_WW1 = 'MIX_REMIX_FRIENDS_WW1'  # Remixed value from WW1 mix (only for WW2)
    MIX_STAY = 'MIX_STAY'    # Mix output not yet spend (may be remixed or leave mix later)


class MIX_PROTOCOL(Enum):
    UNSET = 'UNSET'  # not set yet
    WASABI1 = 'WASABI1'  # Wasabi 1.0
    WASABI2 = 'WASABI2'  # Wasabi 2.0
    WHIRLPOOL = 'WHIRLPOOL'  # Whirlpool
    JOINMARKET = 'JOINMARKET'  # JoinMarket


class SummaryMessages:
    summary_messages = []

    def print(self, message: str):
        logging.info(message)
        self.summary_messages.append(message)

    def print_summary(self):
        print(f'Total log messages: {len(self.summary_messages)}')
        for message in self.summary_messages:
            print(message)


class CJ_LOG_TYPES(Enum):
    ROUND_STARTED = 'ROUND_STARTED'
    BLAME_ROUND_STARTED = 'BLAME_ROUND_STARTED'
    COINJOIN_BROADCASTED = 'COINJOIN_BROADCASTED'
    INPUT_BANNED = 'INPUT_BANNED'
    NOT_ENOUGH_FUNDS = 'NOT_ENOUGH_FUNDS'
    NOT_ENOUGH_PARTICIPANTS = 'NOT_ENOUGH_PARTICIPANTS'
    WRONG_PHASE = 'WRONG_PHASE'
    MISSING_PHASE_BY_TIME = 'MISSING_PHASE_BY_TIME'
    SIGNING_PHASE_TIMEOUT = 'SIGNING_PHASE_TIMEOUT'
    ALICE_REMOVED = 'ALICE_REMOVED'
    FILLED_SOME_ADDITIONAL_INPUTS = 'FILLED_SOME_ADDITIONAL_INPUTS'
    UTXO_IN_PRISON = 'UTXO_IN_PRISON'


class CJ_ALICE_TYPES(Enum):
    ALICE_REGISTERED = 'ALICE_REGISTERED'
    ALICE_CONNECTION_CONFIRMED = 'ALICE_CONNECTION_CONFIRMED'
    ALICE_READY_TO_SIGN = 'ALICE_READY_TO_SIGN'
    ALICE_POSTED_SIGNATURE = 'ALICE_POSTED_SIGNATURE'


class PRECOMP_STRPTIME:
    precomp_strptime = {}
    precomp_strftime = {}

    def strptime(self, datestr: str, datestr_format: str) -> datetime:
        if datestr not in self.precomp_strptime:
            self.precomp_strptime[datestr] = datetime.strptime(datestr, datestr_format)
        return self.precomp_strptime[datestr]

    def fromisoformat(self, datestr: str) -> datetime:
        if datestr not in self.precomp_strptime:
            self.precomp_strptime[datestr] = datetime.fromisoformat(datestr)
        return self.precomp_strptime[datestr]

    def strftime(self, dt: datetime) -> str:
        if dt not in self.precomp_strftime:
            self.precomp_strftime[dt] = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        return self.precomp_strftime[dt]


class CoinMixInfo:
    num_coins = 0
    num_mixes = 0
    pool_size = -1

    def clear(self):
        self.num_coins = 0
        self.num_mixes = 0
        self.pool_size = -1


class CoinJoinStats:
    pool_100k = CoinMixInfo()
    pool_1M = CoinMixInfo()
    pool_5M = CoinMixInfo()
    no_pool = CoinMixInfo()
    cj_type = MIX_PROTOCOL.UNSET

    def clear(self):
        self.pool_100k.clear()
        self.pool_100k.pool_size = 100000
        self.pool_1M.clear()
        self.pool_1M.pool_size = 1000000
        self.pool_5M.clear()
        self.pool_5M.pool_size = 5000000
        self.no_pool.clear()

        self.cj_type = MIX_PROTOCOL.UNSET


class CJ_TX_CHECK(IntFlag):
    NONE       = 0
    CORE_STRUCTURE              = auto()
    INOUTS_RATIO_THRESHOLD      = auto()
    NUM_INOUT_THRESHOLD         = auto()
    ADDRESS_REUSE_THRESHOLD     = auto()
    MIN_SAME_VALUES_THRESHOLD   = auto()
    OP_RETURN                   = auto()
    MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS = auto()
    # Basic always enforced rules
    BASIC = CORE_STRUCTURE | INOUTS_RATIO_THRESHOLD | NUM_INOUT_THRESHOLD | ADDRESS_REUSE_THRESHOLD | OP_RETURN | MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS
    # All defined rules
    ALL = BASIC | MIN_SAME_VALUES_THRESHOLD

CJ_TX_CHECK_JOINMARKET_DEFAULTS = {
    CJ_TX_CHECK.INOUTS_RATIO_THRESHOLD: 4,      # 4 times more of inputs or outputs
    CJ_TX_CHECK.NUM_INOUT_THRESHOLD: 100,       # More than 100 inputs or outputs
    CJ_TX_CHECK.ADDRESS_REUSE_THRESHOLD: 0.34,  # More than 34% of address reuse
    CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD: 2,   # At least two same outputs
    CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS: 1, # How many different equal outputs are allowed
    CJ_TX_CHECK.OP_RETURN: {''}                 # Any OP_RETURN without any additional check
}

CJ_TX_CHECK_WHIRLPOOL_DEFAULTS = {}
CJ_TX_CHECK_WASABI1_DEFAULTS = {}
CJ_TX_CHECK_WASABI2_DEFAULTS = {}

class FILTER_REASON(Enum):
    VALID = 'VALID'
    UNSPECIFIED = 'UNSPECIFIED'
    INVALID_STRUCTURE = 'INVALID_STRUCTURE'
    INOUTS_RATIO_THRESHOLD = 'INOUTS_RATIO_THRESHOLD'
    NUM_INOUT_THRESHOLD = 'NUM_INOUT_THRESHOLD'
    ADDRESS_REUSE_THRESHOLD = 'ADDRESS_REUSE_THRESHOLD'
    MIN_SAME_VALUES_THRESHOLD = 'MIN_SAME_VALUES_THRESHOLD'
    OP_RETURN = 'OP_RETURN'
    MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS = 'MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS'


class ClusterIndex:
    NEW_CLUSTER_INDEX = 0

    def __init__(self, initial_cluster_index):
        self.NEW_CLUSTER_INDEX = initial_cluster_index

    def get_new_index(self):
        self.NEW_CLUSTER_INDEX += 1
        return self.NEW_CLUSTER_INDEX

    def get_current_index(self):
        return self.NEW_CLUSTER_INDEX


# Global object for computation of clusters (possibly refactor)
CLUSTER_INDEX = ClusterIndex(0)

# Global object for important messages
SM = SummaryMessages()

# Global object for precomputated datetime -> datetime string
precomp_datetime = PRECOMP_STRPTIME()
