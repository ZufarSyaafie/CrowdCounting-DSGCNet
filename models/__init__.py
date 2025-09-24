from .DSGCnet import build


def build_model(args, training=False):
    return build(args, training)
