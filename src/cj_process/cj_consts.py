
SATS_IN_BTC = 100000000
MAX_SATS = 21000000*SATS_IN_BTC

VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'


WHIRLPOOL_POOL_NAMES_ALL = ['whirlpool_100k', 'whirlpool_1M', 'whirlpool_5M', 'whirlpool_50M', 'whirlpool_ashigaru_25M',
         'whirlpool_ashigaru_2_5M', 'whirlpool']
WHIRLPOOL_POOL_SIZES = {'whirlpool_100k': 100000, 'whirlpool_1M': 1000000, 'whirlpool_5M': 5000000, 'whirlpool_50M': 50000000,
                        'whirlpool_ashigaru_25M': 25000000, 'whirlpool_ashigaru_2_5M': 2500000}
WHIRLPOOL_FUNDING_TXS = {}
WHIRLPOOL_FUNDING_TXS[100000] = {'start_date': '2021-03-05 23:50:59.000', 'funding_txs': ['ac9566a240a5e037471b1a58ea50206062c13e1a75c0c2de3f21c7053573330a']}
WHIRLPOOL_FUNDING_TXS[1000000] = {'start_date': '2019-05-23 20:54:27.000', 'funding_txs': ['c6c27bef217583cca5f89de86e0cd7d8b546844f800da91d91a74039c3b40fba', 'a42596825352055841949a8270eda6fb37566a8780b2aec6b49d8035955d060e', '4c906f897467c7ed8690576edfcaf8b1fb516d154ef6506a2c4cab2c48821728']}
WHIRLPOOL_FUNDING_TXS[5000000] = {'start_date': '2019-04-17 16:20:09.000', 'funding_txs': ['a554db794560458c102bab0af99773883df13bc66ad287c29610ad9bac138926', '792c0bfde7f6bf023ff239660fb876315826a0a52fd32e78ea732057789b2be0', '94b0da89431d8bd74f1134d8152ed1c7c4f83375e63bc79f19cf293800a83f52', 'e04e5a5932e8d42e4ef641c836c6d08d9f0fff58ab4527ca788485a3fceb2416']}
WHIRLPOOL_FUNDING_TXS[50000000] = {'start_date': '2019-08-02 17:45:23.000', 'funding_txs': ['b42df707a3d876b24a22b0199e18dc39aba2eafa6dbeaaf9dd23d925bb379c59']}
WHIRLPOOL_FUNDING_TXS[2500000] = {'start_date': '2025-05-31 11:16:05.000', 'funding_txs': ['737a867727db9a2c981ad622f2fa14b021ce8b1066a001e34fb793f8da833155']}
WHIRLPOOL_FUNDING_TXS[25000000] = {'start_date': '2025-06-06 01:32:04.000', 'funding_txs': ['7784df1182ab86ee33577b75109bb0f7c5622b9fb91df24b65ab2ab01b27dffa']}


WASABI2_FUNDING_TXS = {}
WASABI2_FUNDING_TXS['zksnacks'] = {'start_date': '2022-06-18 01:38:00.000', 'funding_txs': ['d31c2b4d71eb143b23bb87919dda7fdfecee337ffa1468d1c431ece37698f918']}
WASABI2_FUNDING_TXS['kruw.io'] = {'start_date': '2024-05-18 00:06:06.000', 'funding_txs': ['1be2abf3434a74c3fa76f6b24294fa9ce7cc6afc3a741ee4332c48da657784ac', 'f861aa534a5efe7212a0c1bdb61f7a581b0d262452a79e807afaa2d20d73c8f5', 'b5e839299bfc0e50ed6b6b6c932a38b544d9bb6541cd0ab0b8ddcc44255bfb78']}
WASABI2_FUNDING_TXS['gingerwallet'] = {'start_date': '2024-06-02 18:20:36.000', 'funding_txs': ['75d060816ca08d067a91ba982e330aba7c5a2d50db2605403567989370120a66', 'f861aa534a5efe7212a0c1bdb61f7a581b0d262452a79e807afaa2d20d73c8f5', 'b5e839299bfc0e50ed6b6b6c932a38b544d9bb6541cd0ab0b8ddcc44255bfb78']}
WASABI2_FUNDING_TXS['opencoordinator'] = {'start_date': '2025-05-02 21:43:13.000', 'funding_txs': ['9a15e204577d2a7c7c1861d2f9225a24add5cbdb64ade6c9b90bc2f9a6f21260', 'f861aa534a5efe7212a0c1bdb61f7a581b0d262452a79e807afaa2d20d73c8f5', 'b5e839299bfc0e50ed6b6b6c932a38b544d9bb6541cd0ab0b8ddcc44255bfb78']}

WASABI2_COORD_NAMES_ALL = ["kruw", "gingerwallet", "opencoordinator", "wasabicoordinator",
                  "coinjoin_nl", "wasabist", "dragonordnance", "mega", "btip",
                  "strange_2025", "unknown_2024_e85631", "unknown_2024_28ce7b", "others", "zksnacks"]

WASABI1_COORD_NAMES_ALL = ["others", "zksnacks"]

# Threshold from BIP-125: any input with nSequence < 0xFFFFFFFE signals RBF.
RBF_THRESHOLD = 0xFFFFFFFE  # 4294967294

SAFE_CPU_CORES = 24