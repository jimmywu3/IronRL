@echo off
setlocal

echo ===========================================
echo    MineRL Installer for Windows
echo ===========================================

:: 1. Verify Backup File Exists
if not exist "backup_stuff\build.gradle" (
    echo [ERROR] Could not find 'backup_stuff\build.gradle'
    echo Please ensure the backup_stuff folder exists in the root directory.
    pause
    exit /b 1
)

:: 2. Check for Java 8
if "%JAVA_HOME%"=="" (
    echo [ERROR] JAVA_HOME is not set.
    echo Please set JAVA_HOME to your Java 8 installation.
    pause
    exit /b 1
)
echo [OK] Using Java: %JAVA_HOME%

:: 3. Clean and Create Temp Directory
set "BUILD_DIR=temp_minerl_install"
if exist "%BUILD_DIR%" (
    echo Removing old temp directory...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

:: 4. Clone MineRL
echo [INFO] Cloning MineRL v0.4.4...
git clone --depth 1 --branch v0.4.4 https://github.com/minerllabs/minerl.git . >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git clone failed. Check internet connection.
    cd ..
    pause
    exit /b 1
)

:: 5. Replace build.gradle with Backup
echo [INFO] Replacing build.gradle from backup_stuff...
copy /Y "..\backup_stuff\build.gradle" "minerl\Malmo\Minecraft\build.gradle"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy backup file.
    cd ..
    pause
    exit /b 1
)

:: 6. Install
echo [INFO] Installing MineRL... (This will take 5-15 minutes, please be patient)
echo        Ignore "Building wheel" if it seems stuck. It is downloading Minecraft.
python -m pip install .
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed.
    cd ..
    pause
    exit /b 1
)

:: 7. Cleanup
cd ..
echo [INFO] Cleaning up temp files...
rmdir /s /q "%BUILD_DIR%"

echo ===========================================
echo [SUCCESS] MineRL installed successfully.
echo ===========================================
pause