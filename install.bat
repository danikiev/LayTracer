:: Windows installer script for LayTracer
:: This script creates a Conda environment based on the "environment.yml" and installs the package
:: Run: install.bat from cmd
:: D. Anikiev, 2026-02-17

@echo off

set ENV_NAME=laytracer
set PACKAGE_NAME=laytracer
set ENV_YAML=environment.yml

:: Check for Conda Installation
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not found in the system PATH.
    echo Please install Conda and make sure it's added to the system PATH.
    pause
    exit /b 2
)

:: Check for environment file
if not exist %ENV_YAML% (
    echo %ENV_YAML% not found in the current directory
    echo Please check and try again.
    pause
    exit /b 3
)

echo Creating Conda environment %ENV_NAME% from %ENV_YAML%...
call conda env create -f %ENV_YAML%
:: Check if environment creation was successful
if %ERRORLEVEL% equ 0 (
    echo Conda environment %ENV_NAME% created successfully.
) else (
    echo Failed to create Conda environment %ENV_NAME%. Please check the error messages above.
    pause
    exit /b 4
)

:: List conda environments
call conda env list

:: Activate environment
echo Activating Conda environment %ENV_NAME%...
call conda activate %ENV_NAME%
:: Check if environment activation was successful
if %ERRORLEVEL% equ 0 (    
    echo Conda environment %ENV_NAME% activated successfully.
) else (
    echo Failed to activate Conda environment %ENV_NAME%. Please check the error messages above.
    pause
    exit /b 5
)

echo Installing %PACKAGE_NAME%...
call pip install -e .
if %ERRORLEVEL% equ 0 (    
    echo Successfully installed %PACKAGE_NAME%.
) else (
    echo Failed to install %PACKAGE_NAME%. Please check the error messages above.
    pause
    exit /b 6
)

echo Python version:
call python --version

echo Python path:
:: Pick only first output
for /f "tokens=* usebackq" %%f in (`where python`) do (set "pythonpath=%%f" & goto :next)
:next

echo %pythonpath%

echo Done!

pause
