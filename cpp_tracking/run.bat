@echo off
setlocal EnableExtensions

REM ================================
REM Resolve script directory
REM ================================
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM ================================
REM Resolve APP_EXE safely
REM ================================
set "APP_EXE="

if defined TRACKING_APP_EXE (
set "APP_EXE=%TRACKING_APP_EXE%"
)

if not defined APP_EXE (
call :find_app_exe APP_EXE
)

if not defined APP_EXE (
echo [ERROR] tracking_app executable not found.
echo [HINT] Build first:
echo        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="D:/opencv/build/x64/vc16/lib" -DONNXRUNTIME_INCLUDE_DIR="D:/onnxruntime-win-x64-1.25.0/include" -DONNXRUNTIME_LIBRARY="D:/onnxruntime-win-x64-1.25.0/lib/onnxruntime.lib"
echo        cmake --build build --config Release
echo [HINT] Or set TRACKING_APP_EXE to absolute executable path.
exit /b 1
)

REM ================================
REM Resolve config file
REM ================================
set "CONFIG_FILE=%~1"
if not defined CONFIG_FILE set "CONFIG_FILE=%SCRIPT_DIR%\config\app.yaml"

if not exist "%CONFIG_FILE%" (
echo [ERROR] Config file not found: %CONFIG_FILE%
echo [HINT] Usage: run.bat [path-to-app.yaml]
exit /b 1
)

REM ================================
REM Get APP_DIR safely
REM ================================
for %%I in ("%APP_EXE%") do set "APP_DIR=%%~dpI"
if "%APP_DIR:~-1%"=="" set "APP_DIR=%APP_DIR:~0,-1%"

REM ================================
REM Update PATH safely
REM ================================
call :prepend_path_if_exists "%APP_DIR%"

if defined ONNXRUNTIME_LIB_DIR (
call :prepend_path_if_exists "%ONNXRUNTIME_LIB_DIR%"
) else (
if defined ONNXRUNTIME_ROOT call :prepend_path_if_exists "%ONNXRUNTIME_ROOT%\lib"
)

if defined OPENCV_BIN (
call :prepend_path_if_exists "%OPENCV_BIN%"
) else (
if defined OpenCV_DIR (
for %%I in ("%OpenCV_DIR%..\bin") do call :prepend_path_if_exists "%%~fI"
)
if defined OPENCV_ROOT (
call :prepend_path_if_exists "%OPENCV_ROOT%\bin"
call :prepend_path_if_exists "%OPENCV_ROOT%\x64\vc17\bin"
call :prepend_path_if_exists "%OPENCV_ROOT%\x64\vc16\bin"
)
call :prepend_path_if_exists "D:\opencv\build\x64\vc17\bin"
call :prepend_path_if_exists "D:\opencv\build\x64\vc16\bin"
)

echo [INFO] Executable: "%APP_EXE%"
echo [INFO] Config: "%CONFIG_FILE%"

REM ================================
REM Run app
REM ================================
pushd "%SCRIPT_DIR%"
"%APP_EXE%" "%CONFIG_FILE%"
set "APP_EXIT=%ERRORLEVEL%"
popd

if not "%APP_EXIT%"=="0" (
echo [ERROR] tracking_app exited with code %APP_EXIT%
)

exit /b %APP_EXIT%

REM ================================
REM FUNCTIONS
REM ================================

:find_app_exe
set "%~1="
for %%P in (
"%SCRIPT_DIR%\build\bin\Release\tracking_app.exe"
"%SCRIPT_DIR%\build\bin\tracking_app.exe"
"%SCRIPT_DIR%\build\Release\tracking_app.exe"
"%SCRIPT_DIR%\build\tracking_app.exe"
) do (
if exist "%%~fP" (
set "%~1=%%~fP"
goto :eof
)
)
goto :eof

:prepend_path_if_exists
if not exist "%~1" goto :eof

REM tránh duplicate PATH (optional nhưng tốt)
echo %PATH% | find /I "%~1" >nul
if %ERRORLEVEL%==0 goto :eof

set "PATH=%~1;%PATH%"
goto :eof
