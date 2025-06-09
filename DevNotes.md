## Install locally
pip install -r requirements.txt

pip uninstall eegdash -y
python -m pip install --editable /Users/arno/Python/EEG-Dash-Data
# Warning use the exact command above, pip install by itself might not work

### check if working from different folders
python -c "from eegdash import EEGDashDataset; print(EEGDashDataset)"

## Run hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files

## Create package and release on Pypi
Documentation is at https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Update version in pyproject.toml
- Run "python -m build"
- "python -m twine upload --repository testpypi dist/*" OR "python -m twine upload dist/*"
Look for API token in email (different for test and regular)

## Populate database
- Log on mongodb.com with sccn user sccn3709@gmail.com (see email for pass)
- Change eegdash or eegdashstaging in main.py
- Run script/data_ingest.py

# Remount
sudo sshfs -o allow_other,IdentityFile=/home/dung/.ssh/id_rsa arno@login.expanse.sdsc.edu:/expanse/projects/nemar /mnt/nemar/