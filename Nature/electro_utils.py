import pickle
import warnings
import logging
import numpy as np

from typing import Tuple, Type
from pynwb.icephys import PatchClampSeries, CurrentClampStimulusSeries, IntracellularElectrode

from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
from ipfx.data_set_features import extract_data_set_features
from pathlib import Path
from tqdm.notebook import tqdm

from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.dataset.ephys_nwb_data import EphysNWBData
from ipfx.dataset.hbg_nwb_data import HBGNWBData
from ipfx.stimulus import StimulusOntology
from typing import Optional, Dict, Any
from ipfx.dataset.create import get_nwb_version, is_file_mies, LabNotebookReaderIgorNwb, MIESNWBData
from ipfx.sweep import Sweep
import allensdk.core.json_utilities as ju

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.colors import to_hex


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
    def _get_series(self, sweep_number: int,
                    series_class: Tuple[Type[PatchClampSeries]]):
        """Catch IZeroClampStimulusSeries TypeError"""
        try:
            series = super()._get_series(sweep_number, series_class)
            self._set_example_series(series)
            return series
        except TypeError:
            return self._get_izero_current_stimulus_series(sweep_number)

    def _set_example_series(self, series):
        if not hasattr(self, 'example_series'):
            setattr(self, 'example_series', series)

    def _get_izero_current_stimulus_series(self, sweep_number):
        example_series = getattr(self, 'example_series', None)
        if example_series:
            return CurrentClampStimulusSeries(
                name= example_series.name[:-3]+'IZero',
                data=np.zeros(example_series.data[:].shape),
                electrode=example_series.electrode,
                gain=example_series.gain,
                stimulus_description=example_series.stimulus_description,
                resolution=example_series.resolution,
                conversion=example_series.conversion,
                timestamps=example_series.timestamps,
                starting_time=example_series.starting_time,
                rate=example_series.rate,
                comments=example_series.comments,
                description=example_series.description,
                control=example_series.control,
                control_description=example_series.control_description,
                sweep_number=sweep_number,
                unit=example_series.unit
            )
        else:
            return None



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


class CustomEphysDataSet(EphysDataSet):
    def sweep(self, sweep_number: int) -> Sweep:
        sweep_data = self.get_sweep_data(sweep_number)
        sweep_metadata = self._data.get_sweep_metadata(sweep_number)

        time = np.arange(
            len(sweep_data["stimulus"])
        ) / sweep_data["sampling_rate"]

        voltage, current = type(self)._voltage_current(
            sweep_data["stimulus"],
            sweep_data["response"],
            sweep_metadata["clamp_mode"],
            enforce_equal_length=False,
        )

        sample_sizes = {len(time), len(voltage), len(current)}
        if len(sample_sizes)>1:
            sz = min(sample_sizes)
            time = time[:sz]
            voltage = voltage[:sz]
            current = current[:sz]

        try:
            sweep = Sweep(
                t=time,
                v=voltage,
                i=current,
                sampling_rate=sweep_data["sampling_rate"],
                sweep_number=sweep_number,
                clamp_mode=sweep_metadata["clamp_mode"],
                epochs=sweep_data.get("epochs", None),
            )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep


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

    return CustomEphysDataSet(
        sweep_info=sweep_info,
        data=nwb_data,
    )


def plot_iclamp_sweep(dataset, sweep_numbers, highlight_ind, fig:Figure=None, min_max_i=None, min_max_v=None):
    highlight_color='#0779BE'
    background_color='#dddddd'
    highlight_zorder = Line2D.zorder+0.5
    background_zorder = Line2D.zorder

    if fig:
        ax1, ax2 = fig.get_axes()
        def change_lines_color_zorder(ax):
            line_c = set()
            for line in ax.get_lines():
                label = line.get_label()
                c = to_hex(line.get_color())
                if label==f'sweep_{sweep_numbers[highlight_ind]}':
                    line.set_color(highlight_color)
                    line_c|= {highlight_color.lower()}
                    line.set_zorder(highlight_zorder)
                elif c==highlight_color.lower():
                    line.set_color(background_color)
                    line_c|= {background_color.lower()}
                    line.set_zorder(background_zorder)
                else:
                    line_c|= {c}
            return line_c
        vcolors=change_lines_color_zorder(ax1)
        icolors=change_lines_color_zorder(ax2)
        if len(vcolors)<2 and len(icolors)<2:
            return None
        else:
            return fig

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(212)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    def plot_sweep(sweep_number, color=background_color, zorder=background_zorder):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sweep = dataset.sweep(sweep_number)
        ax1.plot(sweep.t, sweep.v*1e-3, color, label=f'sweep_{sweep_number}', zorder=zorder)
        ax2.plot(sweep.t, sweep.i*1e-12, color, label=f'sweep_{sweep_number}', zorder=zorder)

    for sn_ind, sn in enumerate(sweep_numbers):
        if sn_ind==highlight_ind:
            plot_sweep(sn, highlight_color, highlight_zorder)
        else:
            plot_sweep(sn)
    if min_max_i:
        ax2.set_ylim(min_max_i)
    if min_max_v:
        ax1.set_ylim(min_max_v)
    return fig

def plot_nwb_trace(nwb: str, cell_id:str, out_dir: Path, min_max_i=None, min_max_v=None):
    nwb_path = Path(nwb)
    out_path = out_dir/cell_id
    out_path.mkdir(exist_ok=True)

    if not nwb_path.exists():
        logging.warning(f'{nwb} not exist. Ignored!')
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset = custom_create_ephys_data_set(nwb_file=nwb)
            sweep_table = dataset.sweep_table
        sweep_numbers = sweep_table['sweep_number'].tolist()

        fig=None
        for hightlight_ind, sweep_number in tqdm(enumerate(sweep_numbers), total=len(sweep_numbers), desc='plot sweep', leave=False):
            out_fig_path = out_path/f'sweep{sweep_number}.pdf'
            if not out_fig_path.exists():
                res_fig = plot_iclamp_sweep(dataset, sweep_numbers, hightlight_ind, fig=fig,
                                            min_max_i=min_max_i, min_max_v=min_max_v)
                if res_fig:
                    plt.savefig(out_fig_path, dpi=600)
                    fig=res_fig
            else:
                continue
        if fig:
            fig.clf()
            plt.close('all')

if __name__ == '__main__':
    test_nwb = '/opt/data/Nature/nwb/sub-mouse-XECLH_ses-20180327-sample-2_slice-20180327-slice-2_cell-20180327-sample-2_icephys.nwb'
    test_data_set = custom_create_ephys_data_set(test_nwb)
    test_sweep_table=test_data_set.sweep_table
    test_fig = plot_iclamp_sweep(test_data_set, test_sweep_table['sweep_number'].tolist(), 5)
    test_fig.show()