from cj_process.parse_cj_logs import analyze_aggregated_coinjoin_stats


def test_analyze_aggregated_coinjoin_stats_skips_empty_experiments(tmp_path):
    experiment_path = tmp_path / "empty-joinmarket"
    analyze_aggregated_coinjoin_stats(
        {
            str(experiment_path): {
                "coinjoins": {},
                "wallets_info": {},
                "analysis": {"path": str(experiment_path)},
            }
        },
        str(tmp_path),
    )
