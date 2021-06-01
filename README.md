# OGB-LSC challenge - MLCollective Team
This repository contains all the experimentation stuff, we have been working on
for the OGB-LSC challenge (HOMO-LUMO gap prediction task).


### How to use?

**Local venv**

- create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```
- install dependencies
  - CPU
```bash
pip install -r requirements.txt -r requirements-cpu.txt
```
  - GPU
```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

**Docker**
TODO

**DVC stuff**
- tell DVC where how to authenticate to Google Cloud Storage (only on private machine; on the VM it should work out-of-the-box)
```bash
export GOOGLE_CLOUD_PROJECT="ogb-team"
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<your email address>/adc.json
```
- pull all files from the MinIO remote (`dvc pull`) 

