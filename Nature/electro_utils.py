import pickle
import warnings
import logging
import numpy as np
import pandas as pd

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
from EphysExtraction import ephys_extractor as efex
from EphysExtraction import ephys_features as ft
from scipy import integrate
from pynwb import NWBHDF5IO


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

def get_time_voltage_current_currindex0(nwb):
    df = nwb.sweep_table.to_dataframe()
    voltage = np.zeros((len(df['series'][0][0].data[:]), int((df.shape[0]+1)/2)))
    time = np.arange(len(df['series'][0][0].data[:]))/df['series'][0][0].rate
    voltage[:, 0] = df['series'][0][0].data[:]
    current_initial = df['series'][1][0].data[12000]*df['series'][1][0].conversion
    curr_index_0 = int(-current_initial/20) # index of zero current stimulation
    current = np.linspace(current_initial, (int((df.shape[0]+1)/2)-1)*20+current_initial, \
                         int((df.shape[0]+1)/2))
    for i in range(curr_index_0):   # Find all voltage traces from minimum to 0 current stimulation
        voltage[:, i+1] = df['series'][0::2][(i+1)*2][0].data[:]
    for i in range(curr_index_0, int((df.shape[0]+1)/2)-1):   # Find all voltage traces from 0 to highest current stimulation
        voltage[:, i+1] = df['series'][1::2][i*2+1][0].data[:]
    voltage[:, curr_index_0] = df.loc[curr_index_0*2][0][0].data[:]    # Find voltage trace for 0 current stimulation
    return time, voltage, current, curr_index_0


def extract_spike_features(time, current, voltage, start=0.1, end=0.7, fil=10):
    """ Analyse the voltage traces and extract information for every spike (returned in df), and information for all the spikes
    per current stimulus magnitude.

    Parameters
    ----------
    time : numpy 1D array of the time (s)
    current : numpy 1D array of all possible current stimulation magnitudes (pA)
    voltage : numpy ND array of all voltage traces (mV) corresponding to current stimulation magnitudes
    start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
    end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
    fil : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)

    Returns
    -------
    df : DataFrame with information for every detected spike (peak_v, peak_index, threshold_v, ...)
    df_related_features : DataFrame with information for every possible used current stimulation magnitude
    """

    df = pd.DataFrame()
    df_related_features = pd.DataFrame()
    for c, curr in enumerate(current):
        current_array = curr * np.ones_like(time)
        start_index = (np.abs(time - start)).argmin()  # Find closest index where the injection current starts
        end_index = (np.abs(time - end)).argmin()  # Find closest index where the injection current ends
        current_array[:start_index] = 0
        current_array[end_index:len(current_array)] = 0
        EphysObject = efex.EphysSweepFeatureExtractor(t=time, v=voltage[:, c], i=current_array, start=start, \
                                                      end=end, filter=fil)
        EphysObject.process_spikes()

        # Adding peak_height (mV) + code for maximum frequency determination (see further)
        spike_count = 0
        if EphysObject._spikes_df.size:
            EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                    EphysObject._spikes_df['threshold_v'].values
            spike_count = EphysObject._spikes_df['threshold_i'].values.size
        df = pd.concat([df, EphysObject._spikes_df], sort=True)

        # Some easily found extra features
        df_features = EphysObject._sweep_features

        # Adding spike count
        df_features.update({'spike_count': spike_count})

        # Adding spike frequency adaptation (ratio of spike frequency of second half to first half)
        SFA = np.nan
        half_stim_index = ft.find_time_index(time, np.float(start + (end - start) / 2))
        if spike_count > 5:  # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here
            # but in the end we only take the trace with the max amount of spikes

            if np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] < half_stim_index) != 0:
                SFA = np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] > half_stim_index) / \
                      np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] < half_stim_index)

        df_features.update({'SFA': SFA})

        # Adding current (pA)
        df_features.update({'current': curr})

        # Adding membrane voltage (mV)
        df_features.update({'resting_membrane_potential': EphysObject._get_baseline_voltage()})

        # Adding voltage deflection to steady state (mV)
        voltage_deflection_SS = ft.average_voltage(voltage[:, c], time, start=end - 0.1, end=end)
        # voltage_deflection_v, voltage_deflection_i = EphysObject.voltage_deflection() # = old way: max deflection
        df_features.update({'voltage_deflection': voltage_deflection_SS})

        # Adding input resistance (MOhm)
        input_resistance = np.nan
        if not ('peak_i' in EphysObject._spikes_df.keys()) and not curr == 0:  # We only calculate input resistances
            # from traces without APs
            input_resistance = (np.abs(voltage_deflection_SS - EphysObject._get_baseline_voltage()) * 1000) / np.abs(
                curr)
            if input_resistance == np.inf:
                input_resistance = np.nan
        df_features.update({'input_resistance': input_resistance})

        # Adding membrane time constant (s) and voltage plateau level for hyperpolarisation paradigms
        # after stimulus onset
        tau = np.nan
        E_plat = np.nan
        sag_ratio = np.nan
        if curr < 0:  # We use hyperpolarising steps as required in the object function to estimate the
            # membrane time constant and E_plateau
            while True:
                try:
                    tau = EphysObject.estimate_time_constant()  # Result in seconds!
                    break
                except TypeError:  # Probably a noisy bump for this trace, just keep it to be np.nan
                    break
            E_plat = ft.average_voltage(voltage[:, c], time, start=end - 0.1, end=end)
            sag, sag_ratio = EphysObject.estimate_sag()
        df_features.update({'tau': tau})
        df_features.update({'E_plat': E_plat})
        df_features.update({'sag_ratio': sag_ratio})

        # For the rebound and sag time we only are interested in the lowest (-200 pA (usually)) hyperpolarisation trace
        rebound = np.nan
        sag_time = np.nan
        sag_area = np.nan

        if c == 0:
            baseline_interval = 0.1  # To calculate the SS voltage
            v_baseline = EphysObject._get_baseline_voltage()

            end_index = ft.find_time_index(time, 0.7)
            if np.flatnonzero(voltage[end_index:, c] > v_baseline).size == 0:  # So perfectly zero here means
                # it did not reach it
                rebound = 0
            else:
                index_rebound = end_index + np.flatnonzero(voltage[end_index:, c] > v_baseline)[0]
                if not (time[index_rebound] > (end + 0.15)):  # We definitely have 150 ms left to calculate the rebound
                    rebound = ft.average_voltage(
                        voltage[index_rebound:index_rebound + ft.find_time_index(time, 0.15), c], \
                        time[index_rebound:index_rebound + ft.find_time_index(time, 0.15)]) - v_baseline
                else:  # Work with whatever time is left
                    if time[-1] == time[index_rebound]:
                        rebound = 0
                    else:
                        rebound = ft.average_voltage(voltage[index_rebound:, c], \
                                                     time[index_rebound:]) - v_baseline

            v_peak, peak_index = EphysObject.voltage_deflection("min")
            v_steady = ft.average_voltage(voltage[:, c], time, start=end - baseline_interval, end=end)

            if v_steady - v_peak < 4:  # The sag should have a minimum depth of 4 mV
                # otherwise we set sag time and sag area to 0
                sag_time = 0
                sag_area = 0
            else:
                # First time SS is reached after stimulus onset
                first_index = start_index + np.flatnonzero(voltage[start_index:peak_index, c] < v_steady)[0]
                # First time SS is reached after the max voltage deflection downwards in the sag
                if np.flatnonzero(voltage[peak_index:end_index, c] > v_steady).size == 0:
                    second_index = end_index
                else:
                    second_index = peak_index + np.flatnonzero(voltage[peak_index:end_index, c] > v_steady)[0]
                sag_time = time[second_index] - time[first_index]
                sag_area = -integrate.cumtrapz(voltage[first_index:second_index, c], time[first_index:second_index])[-1]

        burst_metric = np.nan
        # print(c)
        if spike_count > 5:
            burst = EphysObject._process_bursts()
            if len(burst) != 0:
                burst_metric = burst[0][0]

        df_features.update({'rebound': rebound})
        df_features.update({'sag_time': sag_time})
        df_features.update({'sag_area': sag_area})
        df_features.update({'burstiness': burst_metric})

        df_related_features = pd.concat([df_related_features, pd.DataFrame([df_features])], sort=True)

    return df, df_related_features


def get_spiking_histogram_with_half_rate(nwb_path, t_range=(0.1, 0.7), num_bins=20):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        io_ = NWBHDF5IO(nwb_path, 'r', load_namespaces=True)
        nwb = io_.read()
        time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(nwb)
        res_df, df_related_features = extract_spike_features(time, current, voltage)
    diff_half_max_rates = abs(df_related_features['avg_rate'] - df_related_features['avg_rate'].max() / 2)
    select_sweep_current = df_related_features.loc[
        diff_half_max_rates == diff_half_max_rates.min(), 'current']
    if select_sweep_current.size>1:
        select_sweep_current = select_sweep_current.iloc[0]
    else:
        select_sweep_current = select_sweep_current.squeeze()
    spiking_t = res_df.loc[res_df['peak_i'] == select_sweep_current, 'peak_t'].values
    hist, bin_edges = np.histogram(spiking_t, bins=num_bins, range=t_range)
    # compute ISI
    spiking_t = spiking_t.compress((spiking_t>=t_range[0])&(spiking_t<=t_range[1]))
    isi = np.diff(spiking_t)
    adapt_index = (isi[1:]-isi[:-1])/(isi[1:]+isi[:-1])
    isi_cv = isi.std()/isi.mean()
    return {
        'spiking_t': spiking_t,
        'spiking_hist': hist,
        'isi': isi,
        'adapt_index': adapt_index,
        'abs_AI': abs(adapt_index),
        'avg_adapt_index': adapt_index.mean(),
        'avg_abs_AI': abs(adapt_index).mean(),
        'isi_cv': isi_cv
    }


if __name__ == '__main__':
    fpath = 'nwb/sub-mouse-XECLH_ses-20180327-sample-1_slice-20180327-slice-1_cell-20180327-sample-1_icephys.nwb'
    res = get_spiking_histogram_with_half_rate(fpath)
    print(res)