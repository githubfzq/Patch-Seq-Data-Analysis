from neuron_morphology.feature_extractor.marked_feature import (
    specialize, nested_specialize)
from neuron_morphology.feature_extractor.feature_specialization import (
    AxonSpec, DendriteSpec, AxonCompareSpec, DendriteCompareSpec
)
from neuron_morphology.features.dimension import dimension
from neuron_morphology.features.intrinsic import (
    num_nodes, num_branches, num_tips, mean_fragmentation, max_branch_order
)
from neuron_morphology.features.branching.bifurcations import (
    num_outer_bifurcations, mean_bifurcation_angle_local, mean_bifurcation_angle_remote
)
from neuron_morphology.features.size import (
    total_length, total_surface_area, total_volume, mean_diameter,
    mean_parent_daughter_ratio, max_euclidean_distance
)
from neuron_morphology.features.path import (
    max_path_distance, early_branch_path, mean_contraction
)
from neuron_morphology.features.statistics.overlap import overlap
from neuron_morphology.feature_extractor.mark import RequiresSoma, RequiresReferenceLayerDepths, Mark
from neuron_morphology.features.layer.reference_layer_depths import DEFAULT_MOUSE_ME_MET_REFERENCE_LAYER_DEPTHS
from neuron_morphology.feature_extractor.feature_specialization import FeatureSpecialization

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import numpy as np
import pandas as pd
import warnings
from tqdm.notebook import tqdm
from itertools import cycle

from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.constants import AXON, BASAL_DENDRITE, SOMA
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.features.statistics.coordinates import COORD_TYPE_SPECIALIZATIONS
from neuron_morphology.morphology import Morphology
from neuron_morphology.transforms.affine_transform import (
    rotation_from_angle, affine_from_transform_translation, AffineTransform)

from neurom import load_neuron
from neurom.view import view
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


custom_features = [
    nested_specialize(
        dimension,
        [COORD_TYPE_SPECIALIZATIONS, {AxonSpec}]
    ),
    specialize(num_nodes, {AxonSpec}),
    specialize(num_branches, {AxonSpec}),
    specialize(num_tips, {AxonSpec}),
    specialize(mean_fragmentation, {AxonSpec}),
    specialize(max_branch_order, {AxonSpec}),
    specialize(num_outer_bifurcations, {AxonSpec}),
    specialize(mean_bifurcation_angle_local, {AxonSpec}),
    specialize(mean_bifurcation_angle_remote, {AxonSpec}),
    specialize(total_length, {AxonSpec}),
    specialize(total_surface_area, {AxonSpec}),
    specialize(total_volume, {AxonSpec, DendriteSpec}),
    specialize(mean_diameter, {AxonSpec}),
    specialize(mean_parent_daughter_ratio, {AxonSpec}),
    specialize(max_euclidean_distance, {AxonSpec}),
    max_path_distance,
    early_branch_path,
    mean_contraction,
    nested_specialize(
        overlap,
        [{AxonSpec, DendriteSpec}, {AxonCompareSpec, DendriteCompareSpec}]
    )
]

basal_extractor = FeatureExtractor()
basal_extractor.register_features(custom_features)

# drop axon.mean_fragmentaion
custom_features_1 = custom_features.copy()
custom_features_1.remove(custom_features_1[4])

opt_extractor = FeatureExtractor()
opt_extractor.register_features(custom_features_1)

with open("../morpho_required_features.pkl", 'rb') as _f:
    required_features = pickle.load(_f)


def plot_swc_raw(swc_path):
    """Raw plotting method indicated by neuron_morphology package."""
    morph = morphology_from_swc(swc_path)
    axon_nodes = morph.get_node_by_types([AXON])
    dend_nodes = morph.get_node_by_types([BASAL_DENDRITE])
    soma_nodes = morph.get_node_by_types([SOMA])

    axon_x = [node['x'] for node in axon_nodes]
    axon_y = [node['y'] for node in axon_nodes]

    dend_x = [node['x'] for node in dend_nodes]
    dend_y = [node['y'] for node in dend_nodes]

    soma_x = [node['x'] for node in soma_nodes]
    soma_y = [node['y'] for node in soma_nodes]

    plt.figure(figsize=(10, 10))

    plt.scatter(axon_x, axon_y, s=1, edgecolor="none")
    plt.scatter(dend_x, dend_y, s=1, edgecolor="none")
    plt.scatter(soma_x, soma_y, s=20, c="black", edgecolor="none")

    plt.gca().set(xticks=[], yticks=[])
    plt.gca().set_aspect("equal")
    sns.despine(left=True, bottom=True)


def get_hierarchy_items(dic):
    if isinstance(dic, dict):
        for k, v in dic.items():
            if isinstance(v, dict):
                for sub_k, sub_v in get_hierarchy_items(v):
                    yield (str(k)+'.'+sub_k, sub_v)
            elif isinstance(v, list) or isinstance(v, np.ndarray):
                pass
            else:
                yield (str(k), v)


def get_required_features(results_dic):
    return filter(lambda kv: kv[0] in required_features.tolist(), 
                  get_hierarchy_items(results_dic))


def get_required_features_from_swcs(swc_iter,
                                    extractors=(basal_extractor, opt_extractor)):
    for swc in swc_iter:
        morpho = morphology_from_swc(swc)
        morpho_data = Data(morpho)
        try:
            feature_extraction_run = extractors[0].extract(morpho_data)
        except ZeroDivisionError:
            logging.warning(f'Error extracting {swc}, continue using opt extractor.')
            feature_extraction_run = extractors[1].extract(morpho_data)
        except:
            logging.error(f'Error extracting {swc}, continue next neuron.')
            yield dict()
            continue
        features = dict(get_required_features(feature_extraction_run.results))
        yield features


def add_scalebar(ax, num, unit, fig, bbox_to_anchor=[0,0,1,1], x_label=None, y_label=None):
    """Add scalebar to axes `ax`.
    num: number or tuple (number, number). Scalebar length.
    unit: String, or (Sring, string).
    x_label, y_label: set defined x-label(y-label). Default: num+unit."""
    if isinstance(num, tuple):
        num_x, num_y = num[0], num[1]
    else:
        num_x, num_y = num, num
    if isinstance(num, tuple):
        unit_x, unit_y = unit[0], unit[1]
    else:
        unit_x, unit_y = unit, unit
    x_label = str(num_x)+unit_x if x_label is None else x_label
    y_label = str(num_y)+unit_y if y_label is None else y_label
    axin = zoomed_inset_axes(
        ax,
        num_x,
        "lower right",
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=fig.transFigure,
        axes_kwargs={
            "xlabel": x_label,
            "ylabel": y_label,
            "xticks": [],
            "yticks": [],
        },
    )
    axin.set_ylim(top=num_y/num_x)
    axin.yaxis.set_label_position("right")
    axin.spines["top"].set_visible(False)
    axin.spines["left"].set_visible(False)
    axin.spines["right"].set_visible(True)
    axin.patch.set_alpha(0)


TREE_COLOR = {BASAL_DENDRITE: 'red', AXON: 'blue', SOMA: 'black'}

def plot_swc(swc_path, plot_method='neurom'):
    """plot_method: (neurom, neuron_morphology)"""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    if plot_method=='neurom':
        morph = load_neuron(swc_path)
        view.plot_neuron(ax, morph)
    elif plot_method=='neuron_morphology':
        morph = morphology_from_swc(swc_path)
        plot_morphology(ax, morph)
    else:
        raise ValueError('plot_method should be in ("neurom", "neuron_morphology").')
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_axis_off()
    add_scalebar(ax, 100, "$\mu m$", fig, [0,0.1,0.9,0.9])
    return fig


def plot_morphology(ax, morph):
    soma = morph.get_soma()
    # plot segmentations
    segmentations = morph.get_segment_list()
    segs = [[(node['x'], node['y']) for node in seg] for seg in segmentations]
    colors = [TREE_COLOR.get(seg[0]['type']) for seg in segmentations]
    linewidths = [seg[0]['radius'] + seg[1]['radius'] for seg in segmentations]
    collections = LineCollection(segs, colors=colors, linewidths=linewidths)
    ax.add_collection(collections)
    # plot soma
    soma_circle = Circle((soma['x'], soma['y']), soma['radius'], color=TREE_COLOR.get(SOMA))
    ax.add_artist(soma_circle)


def plot_and_save_swcs(to_plot_swcs, fig_path, method='neurom'):
    methods = cycle(['neurom', 'neuron_morphology'])
    cur_method = next(methods)
    while cur_method!=method:
        cur_method=next(methods)
    for swc in tqdm(to_plot_swcs):
        to_save = fig_path/(swc.stem+'.pdf')
        if not to_save.exists():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    fig=plot_swc(swc, cur_method)
                except:
                    cur_method = next(methods)
                    logging.warning(f'Use {cur_method} to plot {swc}.')
                    fig=plot_swc(swc, cur_method)
                    cur_method = next(methods)
            plt.savefig(to_save, dpi=600, transparent=True)
            fig.clf()
            plt.close(fig)


class AboveSomaSpec(FeatureSpecialization):
    name="above_soma"
    marks={RequiresSoma}
    kwargs={'position_against_soma': 'above'}

class BelowSomaSpec(FeatureSpecialization):
    name="below_soma"
    marks={RequiresSoma}
    kwargs={'position_against_soma': 'below'}

class AllLayerSpec(FeatureSpecialization):
    name='all_layers'
    marks={RequiresReferenceLayerDepths}
    kwargs={'filter_layers': list(DEFAULT_MOUSE_ME_MET_REFERENCE_LAYER_DEPTHS.keys())}


def nodes_ratio(data: Data, position_against_soma=None, node_types=None, filter_layers=None):
    """Compute ratio of nodes above or below soma, within each layer."""
    soma = data.morphology.get_soma()
    if position_against_soma=='above':
        criterion=lambda nod:((nod['type'] in node_types) if node_types is not None else True) and nod['y']<soma['y']
    elif position_against_soma=='below':
        criterion=lambda nod:((nod['type'] in node_types) if node_types is not None else True) and nod['y']>soma['y']
    else:
        criterion=lambda nod:nod['type'] in node_types if node_types is not None else True
    num_type_nodes = data.morphology.get_node_by_types(node_types)
    if filter_layers and isinstance(filter_layers, list):
        result = {}
        for layer in filter_layers:
            filter_layer_depth = data.reference_layer_depths.get(layer)
            y_min, y_max, is_scale = filter_layer_depth.pia_side, filter_layer_depth.wm_side, filter_layer_depth.scale
            if is_scale:
                selected_nodes = data.morphology.filter_nodes(lambda nod: criterion(nod) and y_min<nod['y']<y_max)
            else:
                selected_nodes = data.morphology.filter_nodes(lambda nod: criterion(nod) and nod['y']>y_min)
            result[layer]=len(selected_nodes)/len(num_type_nodes)
        return result
    else:
        selected_nodes = data.morphology.filter_nodes(criterion)
        return len(selected_nodes)/len(num_type_nodes)

def length_ratio(data: Data, position_against_soma=None, node_types=None, filter_layers=None):
    """Compute ratio of length against total length of specific node types, above or below soma, within each layer."""
    soma = data.morphology.get_soma()
    if position_against_soma=='above':
        criterion=lambda nod:((nod['type'] in node_types) if node_types is not None else True) and nod['y']<soma['y']
    elif position_against_soma=='below':
        criterion=lambda nod:((nod['type'] in node_types) if node_types is not None else True) and nod['y']>soma['y']
    else:
        criterion=lambda nod:nod['type'] in node_types if node_types is not None else True
    all_compartments = data.morphology.get_compartments(node_types=node_types)
    total_length = sum(data.morphology.get_compartment_length(t) for t in all_compartments
                          if t[0]['type']!=SOMA and data.morphology.parent_of(t[0]))
    if filter_layers and isinstance(filter_layers, list):
        result = {}
        for layer in filter_layers:
            filter_layer_depth = data.reference_layer_depths.get(layer)
            y_min, y_max, is_scale = filter_layer_depth.pia_side, filter_layer_depth.wm_side, filter_layer_depth.scale
            if is_scale:
                selected_nodes = data.morphology.filter_nodes(lambda nod: criterion(nod) and (y_min<nod['y']<y_max))
            else:
                selected_nodes = data.morphology.filter_nodes(lambda nod: criterion(nod) and nod['y']>y_min)
            if selected_nodes:
                selected_compartments = data.morphology.get_compartments(selected_nodes) # selected_nodes not empty!
                selected_length = sum(data.morphology.get_compartment_length(t) for t in selected_compartments
                                  if t[0]['type']!=SOMA and data.morphology.parent_of(t[0]))
            else:
                selected_length = 0
            result[layer]=selected_length/total_length
        return result
    else:
        selected_nodes = data.morphology.filter_nodes(criterion)
        selected_compartments = data.morphology.get_compartments(selected_nodes)
        selected_length = sum(data.morphology.get_compartment_length(t) for t in selected_compartments
                              if t[0]['type']!=SOMA and data.morphology.parent_of(t[0]))
        return selected_length/total_length

layer_features = [
    nested_specialize(nodes_ratio, [{AboveSomaSpec}, {AxonSpec}]),
    nested_specialize(nodes_ratio, [{BelowSomaSpec}, {AxonSpec}]),
    nested_specialize(nodes_ratio, [{AllLayerSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{AboveSomaSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{BelowSomaSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{AllLayerSpec}, {AxonSpec}])
]


def nodes_distribution(data: Data, node_types=None, num_bins=20, use_cortex_depth_range=False):
    """Calculate node numbers of bins across y-axis between morphology range or given range.
    use_cortex_depth_range: Use the data.cortex_depth_range. Default: False, use morphology range."""
    type_nodes = data.morphology.get_node_by_types(node_types)
    type_y_coords = [nod['y'] for nod in type_nodes]
    y_min, y_max = getattr(data, 'cortex_depth_range') if use_cortex_depth_range else (min(type_y_coords), max(type_y_coords))
    y_bins = np.linspace(y_min, y_max, num_bins+1)
    res_cuts = pd.cut(type_y_coords, y_bins, include_lowest=True)
    return res_cuts.value_counts().values/len(type_nodes)

def length_distribution(data: Data, node_types=None, num_bins=20, use_cortex_depth_range=False):
    type_nodes = data.morphology.get_node_by_types(node_types)
    all_compartments = data.morphology.get_compartments(node_types=node_types)
    total_length_ = sum(data.morphology.get_compartment_length(t) for t in all_compartments
                        if t[0]['type']!=SOMA and data.morphology.parent_of(t[0]))
    type_y_coords = [nod['y'] for nod in type_nodes]
    y_min, y_max = getattr(data, 'cortex_depth_range') if use_cortex_depth_range else (min(type_y_coords), max(type_y_coords))
    y_bins = np.linspace(y_min, y_max, num_bins + 1)
    res_cuts = pd.cut(type_y_coords, y_bins, include_lowest=True)
    res_lengths = []
    for cut_ in res_cuts.categories:
        selected_nodes = np.compress(res_cuts==cut_, type_nodes).tolist()
        if selected_nodes:
            selected_compartments = data.morphology.get_compartments(selected_nodes)
            selected_length = sum(data.morphology.get_compartment_length(t) for t in selected_compartments
                                  if t[0]['type']!=SOMA and data.morphology.parent_of(t[0]))
            res_lengths.append(selected_length)
        else:
            res_lengths.append(0)
    return np.array(res_lengths)/total_length_

class RequireCortexDepthRange(Mark):
    @classmethod
    def validate(cls, data: Data) -> bool:
        return hasattr(data, "cortex_depth_range")

class AcrossCortexSpec(FeatureSpecialization):
    name="across_cortex"
    marks={RequireCortexDepthRange}
    kwargs={'use_cortex_depth_range': True, 'num_bins': 500}


distribution_features = [
    specialize(nodes_distribution, {AxonSpec}),
    specialize(length_distribution, {AxonSpec})
]

across_cortex_distribution_features = [
    nested_specialize(length_distribution, [{AcrossCortexSpec}, {AxonSpec}]),
    nested_specialize(nodes_distribution, [{AcrossCortexSpec}, {AxonSpec}])
]


def updown_and_move_soma_to_depth(morphology: Morphology, target_soma_depth, return_transformed=False):
    """
    First make morphology upside down, and translate morphology to specific y with x,z keeping original,
    preparing to analyse layer features.

    :param morphology: Input morphology.
    :param target_soma_depth: The y-coordination that the soma translated to. Useful as the ideal depth of soma from pia.
    """
    soma_morph = morphology.get_soma()
    translation_to_origin = np.array([-soma_morph['x'], -soma_morph['y'], -soma_morph['z']])
    upside_matrix = rotation_from_angle(np.pi, axis=2)
    to_origin_upside_matrix = affine_from_transform_translation(
        transform=upside_matrix, translation=translation_to_origin, translate_first=True)
    final_translation_matrix = affine_from_transform_translation(
        translation=np.array([soma_morph['x'], target_soma_depth, soma_morph['z']]))
    affines = final_translation_matrix.dot(to_origin_upside_matrix)
    transformer = AffineTransform(affines)
    if return_transformed:
        return transformer.transform_morphology(morphology, clone=True)
    else:
        transformer.transform_morphology(morphology)



if __name__ == '__main__':
    from pathlib import Path
    from neuron_morphology.feature_extractor.feature_writer import FeatureWriter
    from functools import partial
    test_swc=Path('swc/643536919_transformed.swc')
    test_morpho = morphology_from_swc(test_swc)
    updown_and_move_soma_to_depth(test_morpho, 500)

    # test_data = Data(test_morpho, reference_layer_depths=DEFAULT_MOUSE_ME_MET_REFERENCE_LAYER_DEPTHS)
    test_data = Data(test_morpho, cortex_depth_range=(0, DEFAULT_MOUSE_ME_MET_REFERENCE_LAYER_DEPTHS['6b'].wm_side))
    test_feature = [
        nested_specialize(length_distribution, [{AcrossCortexSpec}, {AxonSpec}]),
        nested_specialize(nodes_distribution, [{AcrossCortexSpec}, {AxonSpec}])
    ]
    test_extractor = FeatureExtractor()
    test_extractor.register_features(test_feature)
    test_extractor.register_features(distribution_features)
    test_extract_run = test_extractor.extract(test_data)
    # test_writer = FeatureWriter("test_res_h5.h5", "test_res_df.csv")
    # test_writer.add_run("test", test_extract_run.serialize())
    # res_tb=test_writer.build_output_table()
    # res1 = test_extract_run.results['axon.across_cortex.length_distribution']
    # res2 = test_extract_run.results['axon.across_cortex.nodes_distribution']
    # plt.bar(np.arange(len(res1)), res1, width=1, alpha=0.3)
    # plt.bar(np.arange(len(res2)), res2, width=1, alpha=0.3)
    # plt.show()
