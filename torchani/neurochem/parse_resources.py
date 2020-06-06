r"""Parse neurochem resources to be used for loading ANI builtin models or specific modules"""
from enum import Enum
from . import Constants
from pkg_resources import resource_filename

class InfoData(Enum):
    CONSTS = 0
    SAE = 1
    PREFIX = 2
    SIZE = 3
    SPECIES = 4

def parse_info_file(info_file_path):
    # this function returns ALL data stored in a neurochem info file 
    # either directly or indirectly
    data = (get_from_info_file(info_file_path, InfoData(j)) for j in range(5))
    return data

def get_from_info_file(info_file_path, data_id):
    # This function parses either file paths, the ensemble prefix, the
    # ensemble size or a species list from a NC info file
    # the data that can be output from this function is a bit eclectic
    # due to the fact that neurochem info files are a bit messy
    info_file = _resolve_resource_path(info_file_path)
    with open(info_file) as f:
        if data_id is InfoData.SPECIES:
            data = f.readlines()[3].strip()
        else:
            data = f.readlines()[data_id.value].strip()

    if data_id is InfoData.SIZE:
        return data

    if data_id is InfoData.SPECIES:
        return Constants(data).species

    return _resolve_resource_path(data)

def _resolve_resource_path(file_path):
    package_name = '.'.join(__name__.split('.')[:1])
    return resource_filename(package_name, 'resources/' + file_path)
