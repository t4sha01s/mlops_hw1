#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# Run both Flask REST API (port 5000)
# and gRPC server (port 50051) in background
# ==================================================

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

echo "Запуск REST API (Flask) на порту 5000..."
nohup python -u app.py > "${LOG_DIR}/flask.log" 2>&1 &
FLASK_PID=$!

echo "Запуск gRPC сервера (порт 50051)..."
nohup python -u grpc_server.py > "${LOG_DIR}/grpc_server.log" 2>&1 &
GRPC_PID=$!

echo "Запуск Streamlit dashboard (порт 8501)..."
nohup streamlit run dashboard.py --server.port 8501 --server.headless true > "${LOG_DIR}/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

echo ""
echo "REST API запущен (PID=${FLASK_PID}) — http://localhost:5000"
echo "gRPC сервер запущен (PID=${GRPC_PID}) — порт 50051"
echo "Dashboard запущен (PID=${STREAMLIT_PID}) — http://localhost:8501"
echo ""
echo "Логи:"
echo "   ${LOG_DIR}/flask.log"
echo "   ${LOG_DIR}/grpc_server.log" 
echo "   ${LOG_DIR}/streamlit.log"
echo ""
echo "Чтобы остановить сервисы: нажмите Ctrl+C"

# Обрабатываем Ctrl+C, чтобы корректно завершить оба процесса
trap "echo 'Останавливаем сервисы...'; kill ${FLASK_PID} ${GRPC_PID}; wait ${FLASK_PID} ${GRPC_PID} 2>/dev/null; echo 'Готово'; exit 0" SIGINT SIGTERM

# Ждём завершения процессов
wait ${FLASK_PID} ${GRPC_PID}
