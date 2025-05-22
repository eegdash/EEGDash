## Install locally

pip install -r requirements.txt

## Create package and release on Pypi
Documentation is at https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Update version in pyproject.toml
- Run "python -m build"
- "python -m twine upload --repository testpypi dist/*" OR "python -m twine upload dist/*"
Look for API token in email (different for test and regular)

## Populate database
- Log on mongodb.com with sccn user sccn3709@gmail.com (see email for pass)
- Run script/data_ingest.py

# Remount
sudo sshfs -o allow_other,IdentityFile=/home/dung/.ssh/id_rsa arno@login.expanse.sdsc.edu:/expanse/projects/nemar /mnt/nemar/