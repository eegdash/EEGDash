# Install

```
pip install -e ".[docs]"
```

Plotly is bundled with the docs extra and is required for the grouped bubble chart
rendering.

# Build

```
cd docs
make build
```

# Run and update in real time

```
sphinx-autobuild docs docs/_build/html
```

