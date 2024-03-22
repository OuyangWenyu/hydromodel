from spotpy.objectivefunctions import rmse
from hydromodel.models.xaj import xaj
from hydromodel.models.gr4j import gr4j
from hydromodel.models.hymod import hymod

CRITERION_DICT = {
    "RMSE": rmse,
}

MODEL_DICT = {
    "xaj_mz": xaj,
    "xaj": xaj,
    "gr4j": gr4j,
    "hymod": hymod,
}
