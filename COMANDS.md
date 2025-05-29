/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv 
source venv/bin/activate
brew install openblas gfortran

pip install --upgrade pip
pip install -r requirements.txt
pip install genetic-prompt-lab

