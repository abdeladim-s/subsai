#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions

Longer description of this module.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
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


def available_subs_formats():
    """
    Returns available subtitle formats
    from :attr:`pysubs2.FILE_EXTENSION_TO_FORMAT_IDENTIFIER`

    :return: list of subtitle formats
    """
    return list(FILE_EXTENSION_TO_FORMAT_IDENTIFIER.keys())
