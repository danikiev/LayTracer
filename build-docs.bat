:: Windows script for building LayTracer documentation
:: This script uses docs/Makefile to build the sphinx documentation in HMTL and PDF
:: Run: `build-docs.bat` from cmd
:: To build PDF as well use: `build-docs.bat -pdf` from cmd
:: D. Anikiev, 2025-03-08

@echo off

:: Check for the -pdf flag
set BUILD_PDF=0
for %%i in (%*) do (
    if "%%i"=="-pdf" set BUILD_PDF=1
)

cd docs

:: Check if build exists
if exist build (
    echo ##################################################################
    echo Cleaning...
    call make cleanall
    :: Check
    if %ERRORLEVEL% neq 0 (
        echo Error while cleaning!
        exit /b 1
    )
)

:: Make HTML documentation
echo ##################################################################
echo Building HTML...
call make html
:: Check
if %ERRORLEVEL% neq 0 (
    echo Error during build process!
    exit /b 1
)

:: Check if PDF building is enabled
if %BUILD_PDF%==1 (
    :: Make PDF documentation
    echo ##################################################################
    echo Building PDF...
    call make latexpdf
    :: Check
    if %ERRORLEVEL% neq 0 (
        echo Error during build process!
        exit /b 1
    )

    :: Copy PDF to HTML folder
    echo ##################################################################
    if exist build\latex\laytracer.pdf (
        echo Copying PDF to HTML folder...
        copy build\latex\laytracer.pdf build\html\_static\laytracer.pdf
        :: Check
        if %ERRORLEVEL% neq 0 (
            echo Error while copying PDF!
            exit /b 2
        )
    ) else (
        echo PDF file not found!
        exit /b 2
    )
) else (
    echo ##################################################################
    echo Skipping PDF build and copy since -pdf flag is not provided.
)

:: Serve
echo ##################################################################
echo Starting server (please check the server port)...
cd build\html
python -m http.server
:: Check
if %ERRORLEVEL% neq 0 (
    echo Error while running server!
    exit /b 3
)

cd ..\..\..
echo Done!
echo ##################################################################
