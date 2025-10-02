echo "Setup Conda venv"
# Create if it doesn’t exist
conda env create -f traffic.yml --name traffic || \
# Otherwise, update existing one
conda env update -f traffic.yml --name traffic --prune

echo "Installing Sumo"
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update

sudo apt-get install -y sumo sumo-tools sumo-doc
