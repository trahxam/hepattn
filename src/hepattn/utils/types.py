from torch import Tensor

DictTensor = dict[str, Tensor]
NestedDictTensor = dict[str, dict[str, Tensor]]
DoubleNestedDictTensor = dict[str, dict[str, dict[str, Tensor]]]
