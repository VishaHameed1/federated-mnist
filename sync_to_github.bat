@echo off
cls
echo [INFO] Initializing Secure Sync Pipeline...
timeout /t 1 >nul

if not exist .git (
    echo [INFO] Initializing Git Repository...
    git init
)

set repo_url=https://github.com/VishaHameed1/federated-mnist.git

echo [INFO] Staging all changes including Docker configuration...
git add .

echo [INFO] Committing changes with professional logs...
git commit -m "refactor: optimize docker orchestration and implement secure weight aggregation"

git remote remove origin >nul 2>&1
git remote add origin %repo_url%

echo [INFO] Synchronizing local branch with GitHub 'main'...
git branch -M main
git push -u origin main --force

echo.
echo [SUCCESS] Project successfully deployed to GitHub.
pause