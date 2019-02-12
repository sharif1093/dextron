# https://github.com/deepmind/dm_control/issues/58
# https://github.com/deepmind/dm_control/issues/24
# https://github.com/deepmind/dm_control/commit/e374a4a75461d07a80a1316463c45f9ccbb45324


import os
from pathlib import Path
from dm_control.utils import io as resources

def getAssets(root, addr):
    path = os.path.join(root, addr)
    if os.path.isdir(path):
        tree = list(Path(path).rglob("*"))
        file_list = [e for e in tree if not os.path.isdir(e)]
    else:
        file_list = [path]
    
    # Exclude any specific entry that you don't want
    assets = {os.path.relpath(file, root): resources.GetResource(file)
              for file in file_list}
    return assets

root_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.join(root_path, "assets")

# ASSETS = getAssets(home_path ,".")
# print("All assets are here:", ASSETS.keys())

def get_model_and_assets_by_name(name):
    """Returns a tuple containing the model XML string and a dict of assets."""
    assets = getAssets(home_path, os.path.join("mesh", name))
    assets.update(getAssets(home_path, "texture"))
    assets.update(getAssets(home_path, "common"))

    model_res = getAssets(home_path, name+".xml")
    model_str = model_res[name+".xml"]

    return model_str, assets


