import logging
import torch
from src.core import models
from copy import deepcopy
# from src.DeepLabFeatureExtractor import DeepLabFeatureExtractor, load_deeplabv3
from torch import nn
import yaml


MODELS = {'video_encoder': models.VideoEncoder,
          'attention_encoder': models.AttentionEncoder,
          'graph_regressor': models.GraphRegressor,
        }
        #   'seg_model': DeepLabFeatureExtractor}



# class AdaptorLayer(nn.Module):
#     def __init__(self):
#         super(AdaptorLayer, self).__init__()
#         # Define a fully connected layer to reduce 2048 -> 256
#         self.fc = nn.Linear(2048, 256)

    # def forward(self, x):
    #     return self.fc(x)

def build(config: dict,
          logger: logging.Logger,
          device: torch.device,
          config_path: str ) -> dict:

    """
    Builds the models dict

    :param config: dict, model config dict
    :param logger: logging.Logger, custom logger
    :param device: torch.device, device to move the models to
    :return: dictionary containing all the submodules (PyTorch models)
    """
    torch.cuda.empty_cache()
    
    config = deepcopy(config)
    try:
        _ = config.pop('checkpoint_path')
        _ = config.pop('pretrained_path')
    except KeyError:
        pass

    # Create the models
    model = {}
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    print(torch.cuda.is_available())

    config_path = config_path
    with open(config_path) as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    


    # if file_config['layer_number'] == "layer4":
        # model['AdaptorLayer'] = AdaptorLayer()
        # model['AdaptorLayer'] = model['AdaptorLayer'].to(device)
    for model_key in config.keys():
        print (model_key)
        if model_key != 'video_encoder':
            model[model_key] = MODELS[model_key](config=config[model_key]).to(device)
        
    

    logger.info_important('Model is built.')

    return model
