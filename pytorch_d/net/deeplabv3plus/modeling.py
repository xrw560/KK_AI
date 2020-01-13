from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3
from . import mobilenetv2


def _segm_mobilenet(num_classes, output_stride):
    aspp_dilate = [12, 24, 36]

    backbone = mobilenetv2.mobilenet_v2(output_stride=output_stride)

    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def deeplabv3plus_mobilenet(num_classes=21):
    return _segm_mobilenet(num_classes, output_stride=8)
