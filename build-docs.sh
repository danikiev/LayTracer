#!/usr/bin/env bash

# Linux/macOS script for building LayTracer documentation
# Run: ./build-docs.sh
# To build PDF as well use: ./build-docs.sh -pdf
# D. Anikiev, 2026-03-03

set -u

BUILD_PDF=0
for arg in "$@"; do
    if [ "$arg" = "-pdf" ]; then
        BUILD_PDF=1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/docs"

if [ ! -d "$DOCS_DIR" ]; then
    echo "docs directory not found: $DOCS_DIR"
    exit 1
fi

cd "$DOCS_DIR" || exit 1

if [ -d build ]; then
    echo "##################################################################"
    echo "Cleaning..."
    make cleanall
    if [ $? -ne 0 ]; then
        echo "Error while cleaning!"
        exit 1
    fi
fi

echo "##################################################################"
echo "Building HTML..."
make html
if [ $? -ne 0 ]; then
    echo "Error during HTML build process!"
    exit 1
fi

if [ $BUILD_PDF -eq 1 ]; then
    echo "##################################################################"
    echo "Building PDF..."
    make latexpdf
    if [ $? -ne 0 ]; then
        echo "Error during PDF build process!"
        exit 1
    fi

    echo "##################################################################"
    if [ -f build/latex/laytracer.pdf ]; then
        echo "Copying PDF to HTML static folder..."
        cp build/latex/laytracer.pdf build/html/_static/laytracer.pdf
        if [ $? -ne 0 ]; then
            echo "Error while copying PDF!"
            exit 2
        fi
    else
        echo "PDF file not found!"
        exit 2
    fi
else
    echo "##################################################################"
    echo "Skipping PDF build and copy since -pdf flag is not provided."
fi

echo "##################################################################"
echo "Starting server (please check the server port)..."
cd build/html || exit 3
python -m http.server
if [ $? -ne 0 ]; then
    echo "Error while running server!"
    exit 3
fi

echo "Done!"
echo "##################################################################"
