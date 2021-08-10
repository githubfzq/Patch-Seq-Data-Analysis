from Cell.morpho_utils import (AllLayerSpec as _CellAllLayerSpec, nodes_ratio, length_ratio,
                               AboveSomaSpec, BelowSomaSpec)
from neuron_morphology.feature_extractor.marked_feature import nested_specialize
from neuron_morphology.feature_extractor.feature_specialization import AxonSpec

class AllLayerSpec(_CellAllLayerSpec):
    kwargs = {'filter_layers': ['L1', 'L2/3', 'L4', 'L5', 'L6', 'wm']}

layer_features = [
    nested_specialize(nodes_ratio, [{AboveSomaSpec}, {AxonSpec}]),
    nested_specialize(nodes_ratio, [{BelowSomaSpec}, {AxonSpec}]),
    nested_specialize(nodes_ratio, [{AllLayerSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{AboveSomaSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{BelowSomaSpec}, {AxonSpec}]),
    nested_specialize(length_ratio, [{AllLayerSpec}, {AxonSpec}])
]