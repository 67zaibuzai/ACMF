from .register import model_entrypoints, _model_entrypoints
from .register import is_model


def build_model(config, model_name, **kwargs):
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, **kwargs)

