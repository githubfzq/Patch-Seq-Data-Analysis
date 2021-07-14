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

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import numpy as np
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.constants import AXON, BASAL_DENDRITE, SOMA
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.features.statistics.coordinates import COORD_TYPE_SPECIALIZATIONS
COORD_TYPE_SPECIALIZATIONS


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


def plot_swc(swc_path):
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
