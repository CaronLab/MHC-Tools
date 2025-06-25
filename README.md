# MHC Tools

### Operation system requirement
Linux or Windows (via WSL)

### Installation

* WSL
```text
wsl --install -d Ubuntu-20.04

```

* Conda installation
```text
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

```

* Python environment
```text
conda create -n mhctools python=3.10
conda activate mhctools
pip install -r requirements.txt
```

### Execution
```text
python ./run_toolname.py
```