r"""Parse neurochem resources to be used for loading ANI builtin models or specific modules"""
from enum import Enum

class InfoData(Enum):
    CONSTS = 0
    SAE = 1
    PREFIX = 2
    SIZE = 3

def parse_info_file(info_file_path):
    data = (get_from_info_file(info_file_path, InfoLines(j)) for j in range(5))
    return data

def get_from_info_file(info_file_path, data):
    # This function parses either file paths, the ensemble prefix or the
    # ensemble size from a NC info file
    info_file = resolve_resource_path(info_file_path)
    with open(info_file) as f:
        file_path = f.readlines()[data.value].strip()
        if data is not InfoLines.SIZE:
            data = resolve_resource_path(file_path)
    return data

def resolve_resource_path(file_path):
    package_name = '.'.join(__name__.split('.')[:-1])
    return resource_filename(package_name, 'resources/' + file_path)
