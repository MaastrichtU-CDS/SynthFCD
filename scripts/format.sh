#!/bin/bash
echo "Running Ruff..."
ruff check --fix .

echo "Running Docformatter..."
docformatter -r --in-place .

echo "Running Black..."
black .

echo "Checking doc coverage..."
interrogate .