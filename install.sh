mkdir -p ../venv/

# Create Python virtual environment named "chipmunq"
python3.12 -m venv ../venv/chipmunq

# Activate the virtual environment
source ../venv/chipmunq/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install tqec
pip install git+https://github.com/tqec/tqec.git