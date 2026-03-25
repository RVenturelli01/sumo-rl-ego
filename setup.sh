# Exit on error (-e), undefined variable (-u), and failed pipe (-o pipefail)
set -euo pipefail           

# Create the virtual environment if it doesn't exist
python3 -m venv .venv

# Activate the virtual environment
source ".venv/bin/activate" 

# Upgrade pip to the latest version
pip install --upgrade pip   

# Install dependencies from requirements.txt (if present)
pip install -r requirements.txt  

# Install the current project in editable mode
pip install -e .            

# Check if 'sumo' is in PATH, warn if not
command -v sumo >/dev/null || echo "Warning: SUMO not found"  

# Print final instruction message
echo "Done. Activate with: source .venv/bin/activate"        