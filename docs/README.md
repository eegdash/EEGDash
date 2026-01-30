# Install

```
pip install -e ".[docs]"
```

# Build (fast, no examples)

```
cd docs
make html-noplot
```

# Build (full examples)

```
cd docs
make html
```

# Run and update in real time

```
sphinx-autobuild docs docs/_build/html
```
