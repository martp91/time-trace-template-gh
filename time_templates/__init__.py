import os
import time_templates

name = "time_templates"

package_path = os.path.dirname(time_templates.__file__)

parent_path = os.path.abspath(os.path.join(package_path, os.pardir))

with open(os.path.join(parent_path, "setup", "data_directory.txt"), "r") as fl:
    data_path = fl.readline()
