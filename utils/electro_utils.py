import pickle
import warnings
from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
from ipfx.data_set_features import extract_data_set_features
from pathlib import Path

from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.dataset.ephys_nwb_data import EphysNWBData
from ipfx.dataset.hbg_nwb_data import HBGNWBData
from ipfx.stimulus import StimulusOntology
from typing import Optional, Dict, Any
from ipfx.dataset.create import get_nwb_version, is_file_mies, LabNotebookReaderIgorNwb, MIESNWBData
import allensdk.core.json_utilities as ju

with open('../electro_required_features.pkl', 'rb') as _data_f:
    required_features = pickle.load(_data_f)


def get_hierarchy_keys(dic_ls):
    if isinstance(dic_ls, dict):
        for k, v in dic_ls.items():
            if isinstance(v, dict) or isinstance(v, list):
                for sub_k in get_hierarchy_keys(v):
                    yield str(k)+'.'+sub_k
            else:
                yield str(k)
    elif isinstance(dic_ls, list):
        if dic_ls:
            if isinstance(dic_ls[-1], dict) or isinstance(dic_ls[-1], list):
                for sub_k in get_hierarchy_keys(dic_ls[0]):
                    yield '<item>.'+sub_k
            else:
                yield '<item>'


def get_hierarchy_items(dic):
    if isinstance(dic, dict):
        for k, v in dic.items():
            if isinstance(v, dict):
                for sub_k, sub_v in get_hierarchy_items(v):
                    yield (str(k)+'.'+sub_k, sub_v)
            elif isinstance(v, list):
                pass
            else:
                yield (str(k), v)


def get_chain_features(total_results, chain_feature):
    res = total_results
    for k in chain_feature.split('.'):
        if res is not None:
            res = res.get(k)
        else:
            break
    return res


def get_required_features(cell_record, cell_features):
    res = dict()
    if cell_record:
        for cr in required_features['cell_records']:
            res[cr] = get_chain_features(cell_record, cr)
    if cell_features:
        for cf in required_features['cell_features']:
            res[cf] = get_chain_features(cell_features, cf)
    return res


def get_required_features_from_nwb(nwb_file, subthresh_min_amp=100):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data_set = create_ephys_data_set(nwb_file=nwb_file)
        drop_failed_sweeps(data_set)
        cell_features, _, cell_record, _, _, _ = extract_data_set_features(
            data_set, subthresh_min_amp=subthresh_min_amp)
        res = get_required_features(cell_record, cell_features)
        return res


class CustomEphysNWBData(EphysNWBData):
    def get_stimulus_unit(self,sweep_number):
        """Catch IZeroClampSeries unit error."""
        try:
            return super(CustomEphysNWBData, self).get_stimulus_unit(sweep_number)
        except TypeError:
            return "None"

class CustomHBGNWBData(CustomEphysNWBData, HBGNWBData):
    def __init__(self,
                 nwb_file: str,
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 validate_stim: bool = True
                 ):
        CustomEphysNWBData.__init__(
            self,
            nwb_file=nwb_file,
            ontology=ontology,
            load_into_memory=load_into_memory,
            validate_stim=validate_stim
        )

def custom_create_ephys_data_set(
        nwb_file: str,
        sweep_info: Optional[Dict[str, Any]] = None,
        ontology: Optional[str] = None
) -> EphysDataSet:
    """
    Overwrite ipfx.dataset.create.create_ephys_data_set().
    """
    nwb_version = get_nwb_version(nwb_file)
    is_mies = is_file_mies(nwb_file)

    if not ontology:
        ontology = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
    if isinstance(ontology, (str, Path)):
        ontology = StimulusOntology(ju.read(ontology))

    if nwb_version["major"] == 2:
        if is_mies:
            labnotebook = LabNotebookReaderIgorNwb(nwb_file)
            nwb_data = MIESNWBData(nwb_file, labnotebook, ontology)
        else:
            nwb_data = CustomHBGNWBData(nwb_file, ontology)

    else:
        raise ValueError(
            "Unsupported or unknown NWB major version {} ({})".format(
                nwb_version["major"], nwb_version["full"]
            )
        )

    return EphysDataSet(
        sweep_info=sweep_info,
        data=nwb_data,
    )
