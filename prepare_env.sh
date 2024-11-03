#!/bin/bash

# Install python pre-installation dependencies
pip install --no-deps -r requirements-preinstall.txt

# Install python static code analysis dependencies
pip install --no-deps -r requirements-static-code-analysis.txt

# Install python dependencies
pip install --no-deps -r requirements.txt

# Verify all dependencies are compatible
pip check

# Install pre-commit
git config --unset-all core.hooksPath
pre-commit install
