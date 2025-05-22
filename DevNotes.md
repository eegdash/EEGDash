## Install locally

pip install -r requirements.txt

## Create package and release on Pypi
Documentation is at https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Update version in pyproject.toml
- Run "python -m build"
- "python -m twine upload --repository testpypi dist/*" OR "python -m twine upload dist/*"
Look for API token in email (different for test and regular)
