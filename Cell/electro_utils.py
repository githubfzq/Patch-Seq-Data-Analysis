import pickle
import warnings
import logging
from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
from ipfx.data_set_features import extract_data_set_features
from pathlib import Path
from tqdm.notebook import tqdm
from ipfx.stim_features import get_stim_characteristics
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
import numpy as np


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


def plot_iclamp_sweep(dataset, sweep_numbers, highlight_ind, show_full_sweep=False, fig:Figure=None,
                      min_max_i=None, min_max_v=None):
    highlight_color='#0779BE'
    background_color='#dddddd'
    highlight_zorder = Line2D.zorder + 0.5
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
        if len(vcolors) < 2 and len(icolors) < 2:
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

    for sn_ind, sn in enumerate(sweep_numbers):
        if sn_ind!=highlight_ind:
            sweep = dataset.sweep(sn)
            ax1.plot(sweep.t, sweep.v, background_color, label=f'sweep_{sn}', zorder=background_zorder)
            ax2.plot(sweep.t, sweep.i, background_color, label=f'sweep_{sn}', zorder=background_zorder)
    highlight_sweep = dataset.sweep(sweep_numbers[highlight_ind])
    ax1.plot(highlight_sweep.t, highlight_sweep.v, highlight_color,
             label=f'sweep_{sweep_numbers[highlight_ind]}', zorder=highlight_zorder)
    ax2.plot(highlight_sweep.t, highlight_sweep.i, highlight_color,
             label=f'sweep_{sweep_numbers[highlight_ind]}', zorder=highlight_zorder)

    if not show_full_sweep:
        stim_start, stim_dur, stim_amp, start_idx, end_idx = get_stim_characteristics(
            highlight_sweep.i, highlight_sweep.t)
        tstart, tend = stim_start-0.05, stim_start+stim_dur+0.05
        ax1.set_xlim(tstart, tend)
        ax2.set_xlim(tstart, tend)
        if not min_max_i:
            auto_adjust_ylim(ax2, dataset, sweep_numbers, start_idx=start_idx, end_idx=end_idx)
        else:
            ax2.set_ylim(min_max_i)
    else:
        if not min_max_i:
            auto_adjust_ylim(ax2, dataset, sweep_numbers)
        else:
            ax2.set_ylim(min_max_i)
    if min_max_v:
        ax1.set_ylim(min_max_v)
    return fig


def auto_adjust_ylim(ax, dataset, sweep_numbers, start_idx=None, end_idx=None):
    """Set ylim to look better."""
    i_ylim_min, i_ylim_max = (None, None)
    for sn_ind, sn in enumerate(sweep_numbers):
        sweep = dataset.sweep(sn)
        lim_Is = sweep.i[start_idx:end_idx]
        if lim_Is.size > 0:
            i_ylim_min = min(lim_Is.min(), i_ylim_min) if i_ylim_min else lim_Is.min()
            i_ylim_max = max(lim_Is.max(), i_ylim_max) if i_ylim_max else lim_Is.max()
    if i_ylim_min and i_ylim_max:
        i_yrange = i_ylim_max - i_ylim_min
        ax.set_ylim(i_ylim_min - i_yrange / 10, i_ylim_max + i_yrange / 10)


def plot_nwb_trace(nwb: str, out_dir: Path, min_max_i=None, min_max_v=None):
    nwb_path = Path(nwb)
    out_path = out_dir/nwb_path.stem
    out_path.mkdir(exist_ok=True)

    if not nwb_path.exists():
        logging.warning(f'{nwb} not exist. Ignored!')
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dataset = create_ephys_data_set(nwb_file=nwb)
            drop_failed_sweeps(dataset)
            sweep_table = dataset.sweep_table
        long_squares_sweep_numbers = sweep_table[(sweep_table['stimulus_name']=='Long Square')&(
            sweep_table['clamp_mode']=='CurrentClamp')]['sweep_number'].tolist()

        fig=None
        for hightlight_ind, sweep_number in tqdm(enumerate(long_squares_sweep_numbers),
                                                 total=len(long_squares_sweep_numbers), desc='plot sweep'):
            out_fig_path = out_path/f'sweep{sweep_number}.pdf'
            if not out_fig_path.exists():
                res_fig = plot_iclamp_sweep(dataset, long_squares_sweep_numbers, hightlight_ind, fig=fig,
                                            min_max_i=min_max_i, min_max_v=min_max_v)
                if res_fig:
                    plt.savefig(out_fig_path, dpi=600, transparent=True)
                    fig=res_fig
        if fig:
            fig.clf()
            plt.close('all')

def get_spiking_histogram_with_half_rate(nwb_path, t_range=(0.5, 1.5), num_bins=50):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dataset = create_ephys_data_set(nwb_file=nwb_path)
        drop_failed_sweeps(dataset)
        cell_features, sweep_features, cell_record, sweep_records, _, _ = extract_data_set_features(
            dataset, subthresh_min_amp=-100
        )
    spike_sweep_features = cell_features['long_squares']['spiking_sweeps']
    sweep_rates = np.array([sweep['avg_rate'] for sweep in spike_sweep_features])
    diff_half_max_rates = abs(sweep_rates-sweep_rates.max()/2)
    select_sweep_id = diff_half_max_rates.argmin()
    spiking_t = np.array([spike['peak_t'] for spike in spike_sweep_features[select_sweep_id]['spikes']])
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
    test_nwb_path = 'nwb/sub-637383947_ses-639389976_icephys.nwb'
    # test_data_set = create_ephys_data_set(nwb_file=test_nwb_path)
    # drop_failed_sweeps(test_data_set)
    # test_sweep_table = test_data_set.sweep_table
    # test_sweep_numbers = test_sweep_table[(test_sweep_table['stimulus_name'] == 'Long Square') & (
    #         test_sweep_table['clamp_mode'] == 'CurrentClamp')]['sweep_number'].tolist()
    # test_cell_features, test_sweep_features, test_cell_record, test_sweep_records, _, _ = extract_data_set_features(
    #     test_data_set, subthresh_min_amp=-100)
    # test_rates = np.array([sweep['avg_rate'] for sweep in test_cell_features['long_squares']['spiking_sweeps']])
    # diff_half_max_rates = abs(test_rates-test_rates.max()/2)
    # select_sweep_id = diff_half_max_rates.argmin()
    # spiking_t = np.array([spike['peak_t']
    #                       for spike in test_cell_features['long_squares']['spiking_sweeps'][select_sweep_id]['spikes']])
    # hist, bin_edges = np.histogram(spiking_t, bins=50, range=(0.5, 1.5))
    res = get_spiking_histogram_with_half_rate(test_nwb_path, (0.5, 1.1), 30)
    print(res)