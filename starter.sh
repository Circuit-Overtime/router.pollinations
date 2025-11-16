MODEL_SERVER_SCRIPT="api/model_server.py"
APP_SCRIPT="api/app.py"
MODEL_PORTS=(7002 7003)
APP_PORT=9000

PIDS=()

echo "[INFO] Starting model servers..."

for PORT in "${MODEL_PORTS[@]}"; do
    echo "[INFO] Launching model server on port $PORT..."
    python3 $MODEL_SERVER_SCRIPT $PORT > logs/model_$PORT.log 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "[INFO] Model server PID: $PID"
done

echo "[INFO] Waiting for model servers to stabilize..."

sleep 2

for i in "${!MODEL_PORTS[@]}"; do
    PORT=${MODEL_PORTS[$i]}
    PID=${PIDS[$i]}

    if ps -p $PID > /dev/null; then
        echo "[INFO] Model server on port $PORT is running (PID $PID)"
    else
        echo "[ERROR] Model server on port $PORT FAILED to start."
        exit 1
    fi
done

echo "[INFO] All model servers are fully online."
echo "[INFO] Starting app server..."

python3 $APP_SCRIPT > logs/app.log 2>&1 &
APP_PID=$!

sleep 1

if ps -p $APP_PID > /dev/null; then
    echo "[INFO] App running on port $APP_PORT (PID $APP_PID)"
else
    echo "[ERROR] App failed to start — check app.log"
    exit 1
fi

echo "[INFO] All services running successfully!"

echo "[INFO] Cleaning up log files..."
for PORT in "${MODEL_PORTS[@]}"; do
    rm -f logs/model_$PORT.log
    echo "[INFO] Removed logs/model_$PORT.log"
done
rm -f app.log
echo "[INFO] Removed logs/app.log"

echo "[INFO] All services running. Press Ctrl+C to stop."

cleanup() {
    echo "[INFO] Shutting down..."
    kill ${PIDS[@]} 2>/dev/null
    kill $APP_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    sleep 5
done