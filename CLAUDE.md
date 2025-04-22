# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Build: `python -m build`
- Install locally: `pip install -e .`
- Test: `ask-bedrock prompt 'How are you doing?'`
- Format: `black ask_bedrock/`
- Type check: `mypy ask_bedrock/`

## Code Style
- **Imports**: Group imports (stdlib, third-party, local)
- **Typing**: Use `collections.abc` types and type hints throughout
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use try/except with the custom `log_error` function
- **Formatting**: Follow PEP 8 guidelines
- **CLI**: Use Click for CLI handling with proper decorations
- **Comments**: Include docstrings for functions and modules

## Project Structure
- Python 3.9+ required
- Configuration in `$HOME/.config/ask-bedrock/config.yaml`
- Main CLI implementation in `ask_bedrock/main.py`
