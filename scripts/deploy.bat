@echo off
setlocal enabledelayedexpansion

set "programName=AlgorithmServer"
set "sourcePath=dist\%programName%"
set "destinationPath=C:\Program Files\GaonPlatform\%programName%"
set "serviceName=GaonPlatform Algorithm Server"
set "anaconda_env=realtime_prediction_server"

echo =======================================================================
echo ================= Starting the deployment process =====================
echo =======================================================================

setlocal enabledelayedexpansion
cd ".."
chcp 65001 > nul 2>&1

rem Step 0: 가상환경 실행 상태 확인
for /f "tokens=4" %%i in ('conda info ^| findstr /C:"active environment"') do (
    set conda_info=%%i
)
set conda_info=!conda_info: =!
echo Current virtual environment: !conda_info!

if "!conda_info!" neq "!anaconda_env!" (
    echo This script must be executed with an active Anaconda virtual environment.
    echo Please activate Anaconda virtual environment and run it again.
    pause
    echo Please press any key to terminate deployment process.
    exit /b 1
)

echo [Anaconda Environment]
for /f "tokens=*" %%a in ('conda info') do (
    echo %%a
)
timeout /t 3 > nul
echo.

rem Step 1: 실행 파일 생성
echo Creating executable file...
pyinstaller main.spec --noconfirm
if %errorlevel% neq 0 (
    echo Error occurred during pyinstaller. Aborting deployment...
    pause
    exit /b %errorlevel%
)
timeout /t 3 > nul
echo.

rem Step 2: 윈도우 서비스 정지
if errorlevel 2 exit
net stop "%serviceName%"
timeout /t 3 > nul

rem Step 3: 서비스 상태 체크 및 대기
:check_service_stopped
sc query "%serviceName%" | find "STOPPED" > nul
if %errorlevel% neq 0 (
    echo %serviceName% is still stopping. Waiting...
    timeout /t 1 > nul
    goto :check_service_stopped
)

rem Step 4: 원본 파일 삭제
:remove_existing
if errorlevel 2 exit

if not exist "%destinationPath%" goto :skip_remove_existing

echo Removing existing %programName%
rmdir /s /q "%destinationPath%" 2>nul || (
    echo Unable to remove %programName%. Make sure it's not in use.
    echo Press any key to retry after closing the program...
    pause > nul
    goto :remove_existing
)
echo %errorlevel%
:skip_remove_existing
timeout /t 3 > nul

rem Step 5: 폴더 이동
echo Moving files to %destinationPath%
robocopy "%sourcePath%" "%destinationPath%" /E /MIR
echo %errorlevel%
if errorlevel 2 (
    echo Unable to copy %programName%. Make sure it's not in use.
    pause
    exit /b 1
)
timeout /t 3 > nul

rem Step 6: 윈도우 서비스 시작
if errorlevel 2 exit
net start "%serviceName%"
if errorlevel 1 (
    echo Error occurred while starting the service. Aborting...
    pause
    exit /b 1
)
sc query "%serviceName%"

echo ========================================================================
echo ============ The deployment has been successfully completed ============
echo ========================================================================

endlocal
