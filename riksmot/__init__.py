import os

pkg_dir, pkg_file = os.path.split(__file__)
DATA_PATH = os.path.join(pkg_dir, "..", "data")

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

NET_PATH = os.path.join(pkg_dir, "..", "networks")

if not os.path.isdir(NET_PATH):
    os.mkdir(NET_PATH)
