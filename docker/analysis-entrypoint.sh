#!/usr/bin/env bash
set -euo pipefail

cd /opt/coinjoin-analysis

show_help() {
  cat <<'EOF'
CoinJoin analysis container

Commands:
  help                         Show this help.
  analyze-emul [args...]       Run python -m cj_process.parse_cj_logs.
  analyze-client [args...]     Run python -m cj_process.ww2_analyze_client.

Any other command is executed as-is, for example:
  docker compose run --rm emulation-process python -m cj_process.parse_cj_logs --help
EOF
}

command_name="${1:-help}"

case "${command_name}" in
  help|-h|--help)
    show_help
    ;;
  analyze-emul)
    shift
    exec python -m cj_process.parse_cj_logs "$@"
    ;;
  analyze-client)
    shift
    exec python -m cj_process.ww2_analyze_client "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
