# Install package if it is missing
def install_import(package_name):
  import os, pkg_resources, importlib
  try:
    pkg_resources.get_distribution(package_name)
  except:
    os.system(f"pip -q install {package_name}")
  pkg = importlib.import_module(package_name)
  return pkg

def install_if_missing(package_name):
  import os, pkg_resources
  try:
    pkg_resources.get_distribution(package_name)
    return
  except:
    os.system(f"pip -q install {package_name}")
    return False

def addpath_if_missing(p):
  import sys
  if p not in sys.path:
    sys.path.append(p)