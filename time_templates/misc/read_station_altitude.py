import os
import xml.etree.ElementTree as ET
from time_templates import package_path
import json

JSON_FILE = os.path.join(package_path, "misc", "map_id_altitude.json")

try:
    with open(JSON_FILE, "r") as infile:
        MAP_ID_ALTITUDE = json.load(infile)
except FileNotFoundError:
    print("Readingt SStationList.xml")
    tree = ET.parse(os.path.join(package_path, "misc", "SStationList.xml"))

    root = tree.getroot()

    MAP_ID_ALTITUDE = {}

    for child in root:
        stationid = int(child.attrib["id"])
        for altitude in child.iter("altitude"):
            altitude = altitude.text
            break
        MAP_ID_ALTITUDE[stationid] = float(altitude)

    with open(JSON_FILE, "w") as out:
        json.dump(MAP_ID_ALTITUDE, out, indent=4)

    print("Mapped id to altitude. Use get_station_height(stationid)")


def get_station_height(stationid):
    try:
        return MAP_ID_ALTITUDE[str(stationid)]
    except KeyError:
        # Assume simulation so 1400m
        return 1400
