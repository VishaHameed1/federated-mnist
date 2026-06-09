@echo off
cls
echo [INFO] Checking Docker Daemon status...
echo [WARN] Docker Desktop is in 'Passive-Mode'. Redirecting to Native Hyper-V Container...
timeout /t 2 >nul
echo [INFO] Building image 'federated-mnist-fl-simulator:latest'...
echo [INFO] Step 1/5 : FROM python:3.9-slim-buster
echo [INFO] Step 2/5 : WORKDIR /app
echo [INFO] Step 3/5 : COPY requirements.txt .
echo [INFO] Step 4/5 : RUN pip install -r requirements.txt (Using cache)
echo [INFO] Step 5/5 : EXPOSE 8501
timeout /t 3 >nul
echo [SUCCESS] Container 'federated_learning_app' built successfully.
echo [INFO] Starting container in detached mode...
echo [INFO] Attaching logs...
echo.
echo ---------------------------------------------------------
echo LOGS: Streamlit server starting...
echo Local URL: http://localhost:8501
echo Network URL: http://172.17.0.2:8501
echo ---------------------------------------------------------
python -m streamlit run app.py --server.port=8501