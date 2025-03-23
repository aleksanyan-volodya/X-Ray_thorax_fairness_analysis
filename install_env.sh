#!/bin/bash
python3 -m venv projet-env
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source projet-env/Scripts/activate  # Windows
else
    source projet-env/bin/activate  # Linux/macOS
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
