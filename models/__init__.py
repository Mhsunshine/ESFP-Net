import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import ResNet50
from models.ESFPNET import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50 , DualStream , C2DResNet50_event, rgb_exactor_seg
# from models.snn_model import Sew_resnet


__factory = {
    'resnet50': ResNet50,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
    'dualstream': DualStream,
    # 'sewresnet': Sew_resnet,
    'rgb_exactor_seg' : rgb_exactor_seg
}


def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        logger.info("Init model: '{}'".format(config.MODEL.NAME))
        if config.MODEL.NAME =='res18' :
            model = __factory[config.MODEL.NAME]()
        else:
            # print(config)
            model = __factory[config.MODEL.NAME](config)
    logger.info("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        # identity_classifier_rgb = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        # identity_classifier_event = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        identity_classifier_rgb = Classifier(feature_dim=2048, num_classes=num_identities)
        identity_classifier_event = Classifier(feature_dim=2048, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    clothes_classifier_rgb = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    
    return model, identity_classifier, clothes_classifier, identity_classifier_rgb, identity_classifier_event