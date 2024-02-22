from pyproj import CRS
from affine import Affine

def jsonify_list(meta_list):
    new = []
    for m in meta_list:
        new.append(jsonify(m))
    return new

def jsonify(meta):
    m = meta.copy()
    m["crs"] = meta["crs"].to_string()
    m["transform"] = tuple(meta["transform"])
    return m

def dejsonify_list(meta_list):
    new = []
    for m in meta_list:
        new.append(dejsonify(m))
    return new

def dejsonify(meta):
    m = meta.copy()
    m["crs"] = CRS(meta["crs"])
    m["transform"] = Affine(*meta["transform"])
    return m
