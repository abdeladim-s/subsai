#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions
"""

import torch
from pysubs2.formats import FILE_EXTENSION_TO_FORMAT_IDENTIFIER


def _load_config(config_name, model_config, config_schema):
    """
    Helper function to load default values if `config_name` is not specified

    :param config_name: the name of the config
    :param model_config: configuration provided to the model
    :param config_schema: the schema of the configuration

    :return: config value
    """
    if config_name in model_config:
        return model_config[config_name]
    return config_schema[config_name]['default']


def get_available_devices() -> list:
    """
    Get available devices (cpu and gpus)

    :return: list of available devices
    """
    return ['cpu', *[f'cuda:{i}' for i in range(torch.cuda.device_count())]]


def available_translation_models() -> list:
    """
    Returns available translation models
    from (dl-translate)[https://github.com/xhluca/dl-translate]

    :return: list of available models
    """
    models = [
        "facebook/m2m100_418M",
        "facebook/m2m100_1.2B",
        "facebook/mbart-large-50-many-to-many-mmt",
    ]
    return models


def available_subs_formats(include_extensions=True):
    """
    Returns available subtitle formats
    from :attr:`pysubs2.FILE_EXTENSION_TO_FORMAT_IDENTIFIER`

    :param include_extensions: include the dot separator in file extensions

    :return: list of subtitle formats
    """

    extensions = list(FILE_EXTENSION_TO_FORMAT_IDENTIFIER.keys())

    if include_extensions:
        return extensions
    else:
        # remove the '.' separator from extension names
        return [ext.split('.')[1] for ext in extensions]
