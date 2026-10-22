#!/usr/bin/env bash
# serve.sh — llama.cpp 决策模型服务管理
# 用法:
#   bash scripts/serve.sh start  [gguf路径]   # 前台服务（Ctrl-C 停）
#   bash scripts/serve.sh daemon [gguf路径]   # 后台服务（nohup + pidfile）
#   bash scripts/serve.sh stop | status | health | logs
#   bash scripts/serve.sh install-systemd [gguf路径]   # 注册 systemd 服务 d3-server
# gguf 缺省取 $D3_WORK/output/ 下最新的 *-Q8_0.gguf
set -euo pipefail
cd "$(dirname "$0")/.." && source ./env.sh

CMD="${1:?用法: serve.sh start|daemon|stop|status|health|logs|install-systemd [gguf]}"
GGUF="${2:-$(ls -t "$D3_WORK"/output/*/d3-*-Q8_0.gguf 2>/dev/null | head -1 || true)}"
PIDFILE="$D3_WORK/llama-server.pid"
LOGFILE="$D3_WORK/llama-server.log"

pick_gguf() {
  [[ -n "$GGUF" && -f "$GGUF" ]] || { echo "[serve] FAIL: 未找到 GGUF（先跑 export.sh 或指定路径）"; exit 1; }
}

case "$CMD" in
  start)
    pick_gguf
    echo "[serve] 前台启动: $GGUF @ port $D3_SERVE_PORT（Ctrl-C 停止）"
    exec env LD_LIBRARY_PATH="$D3_LLAMA_DIR" "$D3_LLAMA_DIR/llama-server" \
      -m "$GGUF" -ngl 99 -c 32768 -ub 2048 -fa on --jinja \
      --port "$D3_SERVE_PORT" --host 0.0.0.0 --api-key "$D3_API_KEY" ;;
  daemon)
    pick_gguf
    [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null && { echo "[serve] 已在运行 PID $(cat $PIDFILE)"; exit 0; }
    nohup env LD_LIBRARY_PATH="$D3_LLAMA_DIR" "$D3_LLAMA_DIR/llama-server" \
      -m "$GGUF" -ngl 99 -c 32768 -ub 2048 -fa on --jinja \
      --port "$D3_SERVE_PORT" --host 0.0.0.0 --api-key "$D3_API_KEY" \
      >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "[serve] 后台启动 PID $(cat $PIDFILE)，日志 $LOGFILE"
    for i in $(seq 1 60); do
      sleep 5
      if curl -sf -m 4 "http://127.0.0.1:$D3_SERVE_PORT/health" >/dev/null 2>&1; then
        echo "[serve] health OK（启动约 $((i*5))s）"; exit 0
      fi
    done
    echo "[serve] WARN: 300s 未就绪，查看日志: tail -30 $LOGFILE"; exit 1 ;;
  stop)
    if [[ -f $PIDFILE ]]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"
      echo "[serve] 已停止"
    else
      pkill -f "llama-server.*-m .*$D3_WORK" 2>/dev/null || pkill -x llama-server 2>/dev/null || true
      echo "[serve] 已清理进程"
    fi ;;
  status)
    if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "running: PID $(cat $PIDFILE)"; else echo "stopped"; fi ;;
  health)
    curl -s -m 5 "http://127.0.0.1:$D3_SERVE_PORT/health" && echo ;;
  logs)
    tail -n "${2:-50}" "$LOGFILE" ;;
  install-systemd)
    pick_gguf
    UNIT=/etc/systemd/system/d3-server.service
    sed -e "s|@USER@|$(whoami)|g" \
        -e "s|@LLAMA@|$D3_LLAMA_DIR|g" \
        -e "s|@GGUF@|$GGUF|g" \
        -e "s|@PORT@|$D3_SERVE_PORT|g" \
        -e "s|@KEY@|$D3_API_KEY|g" \
        -e "s|@WORK@|$D3_WORK|g" \
        "$D3_KIT/scripts/d3-server.service" > /tmp/d3-server.service
    sudo cp /tmp/d3-server.service "$UNIT" && rm /tmp/d3-server.service
    sudo systemctl daemon-reload && sudo systemctl enable --now d3-server
    echo "[serve] systemd 服务已安装并启动: systemctl status d3-server" ;;
  *)
    echo "未知命令: $CMD"; exit 1 ;;
esac
