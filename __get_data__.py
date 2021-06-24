
## Script to extract competition data to disk ##

# packages
import os
from commlit import env_config, competition_download

# set config and data directory
env_config("config.json")
comp = "commonlitreadabilityprize"
comp_dir = os.path.join(os.environ.get("DATA_DIR"), "commonlit-readability")
if not os.path.exists(comp_dir):
    os.makedirs(comp_dir)

# download all data to this directory
competition_download(comp, local_path=comp_dir)
