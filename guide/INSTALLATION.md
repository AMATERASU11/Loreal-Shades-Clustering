# Project Setup Guide — Shades Clustering

This guide explains how to install and use Poetry to manage the dependencies, environments, and development workflow of this project.

## 1. Install Poetry

You must have **Python = 3.11** installed.

Check your Python version:
```
python3 --version
```

If it’s correct, install [Poetry 2.2.0](https://python-poetry.org/docs/#installation) globally:
```
curl -sSL https://install.python-poetry.org | python3 - --version 2.2.0
```

Then make sure Poetry is available in your path:
```
poetry --version
```

If not found, add it manually (depending on your system):
```
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Clone the Project
```
git clone https://github.com/AMATERASU11/Loreal-Shades-Clustering.git
cd Loreal-Shades-Clustering
```

## 3. Run the setup script:
```bash
./setup.sh
```

This will:
- Check for Python 3.11
- Check for Poetry 2.2.0
- Set up a virtual environment

## 4. Install Dependencies

All project dependencies are declared in `pyproject.toml`.

To install everything (main + dev dependencies):
```
poetry install
```

If you only want runtime dependencies (without dev tools like Sphinx or pytest):
```
poetry install --without dev
```

## 5. Use the Virtual Environment

Poetry automatically creates a virtual environment.

To get basic information about the currently activated virtual environment, you can use the `env info` command:
```
poetry env info
```

The `poetry env activate` command prints the activate command of the virtual environment to the console. You can run the output command manually or feed it to the eval command of your shell to activate the environment.
```
eval $(poetry env activate)
```

Now, any Python or Streamlit commands you run will use this environment.

*Alternatively*, you can run commands without entering the shell:
```
poetry run python src/main.py
```

## 6. Running Tests

Tests are located in the `tests/ directory`.
You can run them with:
```
poetry run pytest
```

If you have already activated the environment in your current shell:
```
pytest
```

## 7. Documentation (optional) [Not implemented yet]

If you want to build the Sphinx documentation locally:
```
poetry run sphinx-build -b html docs/ docs/_build/
```

## 8. Adding or Updating Dependencies

To add a new library:
```
poetry add numpy
```

To add a development dependency (like black or pytest):
```
poetry add --group dev black
```

To update dependencies:
```
poetry update
```

## 9. Export Requirements (for non-Poetry users)

If someone prefers using plain pip, they can generate a requirements.txt:
```
poetry export -f requirements.txt --output requirements.txt
```

Then install using:
```
pip install -r requirements.txt
```

## 10. Troubleshooting

Poetry environment not found?
```
poetry env info
poetry env remove python
poetry install
```

To check all available environments:
```
poetry env list
```

To clear all Poetry caches (rarely needed):
```
poetry cache clear pypi --all
```

<hr style="height:7px;border-width:0;color:black;background-color:red">

## Best Practices for Contributors

* Always run `poetry install` after pulling new changes.

* Never manually edit `poetry.lock`.

* Use `poetry add` to install new dependencies.

* Commit both `pyproject.toml` and `poetry.lock`.

* Before pushing, test locally with:
```
poetry run pytest
```
Tests should be running without problems. So for starters, no test has been implemented yet. Therefore, retrieveing 0 element for testing is normal.