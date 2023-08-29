#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SubsAI: Subtitles AI
Subtitles generation tool powered by OpenAI's Whisper and its variants.

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

import os
import pathlib
import tempfile
from typing import Union, Dict

import ffmpeg
import pysubs2
from dl_translate import TranslationModel
from pysubs2 import SSAFile
from subsai.configs import AVAILABLE_MODELS
from subsai.models.abstract_model import AbstractModel
from ffsubsync.ffsubsync import run, make_parser
from subsai.utils import available_translation_models

__author__ = "abdeladim-s"
__contact__ = "https://github.com/abdeladim-s"
__copyright__ = "Copyright 2023,"
__license__ = "GPLv3"
__github__ = "https://github.com/abdeladim/subsai"


class SubsAI:
    """
    Subs AI class

    Example usage:
    ```python
    file = './assets/test1.mp4'
    subs_ai = SubsAI()
    model = subs_ai.create_model('openai/whisper', {'model_type': 'base'})
    subs = subs_ai.transcribe(file, model)
    subs.save('test1.srt')
    ```
    """

    @staticmethod
    def available_models() -> list:
        """
        Returns the supported models

        :return: list of available models
        """
        return list(AVAILABLE_MODELS.keys())

    @staticmethod
    def model_info(model: str) -> dict:
        """
        Returns general infos about the model (brief description and url)

        :param model: model name

        :return: dict of infos
        """
        return {'description': AVAILABLE_MODELS[model]['description'],
                'url': AVAILABLE_MODELS[model]['url']}

    @staticmethod
    def config_schema(model: str) -> dict:
        """
        Returns the configs associated with a model

        :param model: model name

        :return: dict of configs
        """
        return AVAILABLE_MODELS[model]['config_schema']

    @staticmethod
    def create_model(model_name: str, model_config: dict = {}) -> AbstractModel:
        """
        Returns a model instance

        :param model_name: the name of the model
        :param model_config: the configuration dict

        :return: the model instance
        """
        return AVAILABLE_MODELS[model_name]['class'](model_config)

    @staticmethod
    def transcribe(media_file: str, model: Union[AbstractModel, str], model_config: dict = {}) -> SSAFile:
        """
        Takes the model instance (created by :func:`create_model`) or the model name.
        Returns a :class:`pysubs2.SSAFile` <https://pysubs2.readthedocs.io/en/latest/api-reference.html#ssafile-a-subtitle-file>`_

        :param media_file: path of the media file (video/audio)
        :param model: model instance or model name
        :param model_config: model configs' dict

        :return: SSAFile: list of subtitles
        """
        if type(model) == str:
            stt_model = SubsAI.create_model(model, model_config)
        else:
            stt_model = model
        media_file = str(pathlib.Path(media_file).resolve())
        return stt_model.transcribe(media_file)


class Tools:
    """
    Some tools related to subtitles processing (ex: translation)
    """

    def __init__(self):
        pass

    @staticmethod
    def available_translation_models() -> list:
        """
        Returns available translation models
        A simple link to :func:`utils.available_translation_models` for easy access

        :return: list of available models
        """

        return available_translation_models()

    @staticmethod
    def available_translation_languages(model: Union[str, TranslationModel]) -> list:
        """
        Returns the languages supported by the translation model

        :param model: the name of the model
        :return: list of available languages
        """
        if type(model) == str:
            langs = Tools.create_translation_model(model).available_languages()
        else:
            langs = model.available_languages()
        return langs

    @staticmethod
    def create_translation_model(model_name: str = "m2m100", model_family: str = None) -> TranslationModel:
        """
        Creates and returns a translation model instance.

        :param model_name: name of the model. To get available models use :func:`available_translation_models`
        :param model_family: Either "mbart50" or "m2m100". By default, See `dl-translate` docs
        :return: A translation model instance
        """
        mt = TranslationModel(model_or_path=model_name, model_family=model_family)
        return mt

    @staticmethod
    def translate(subs: SSAFile,
                  source_language: str,
                  target_language: str,
                  model: Union[str, TranslationModel] = "m2m100",
                  model_family: str = None,
                  translation_configs: dict = {}) -> SSAFile:
        """
        Translates a subtitles `SSAFile` object, what :func:`SubsAI.transcribe` is returning

        :param subs: `SSAFile` object
        :param source_language: the language of the subtitles
        :param target_language: the target language
        :param model: the translation model, either an `str` or the model instance created by
                        :func:`create_translation_model`
        :param model_family: Either "mbart50" or "m2m100". By default, See `dl-translate` docs
        :param translation_configs: dict of translation configs (see :attr:`configs.ADVANCED_TOOLS_CONFIGS`)

        :return: returns an `SSAFile` subtitles translated to the target language
        """
        if type(model) == str:
            translation_model = Tools.create_translation_model(model_name=model, model_family=model_family)
        else:
            translation_model = model

        translated_subs = SSAFile()
        for sub in subs:
            translated_sub = sub.copy()
            translated_sub.text = translation_model.translate(text=sub.text,
                                                              source=source_language,
                                                              target=target_language,
                                                              batch_size=translation_configs[
                                                                  'batch_size'] if 'batch_size' in translation_configs else 32,
                                                              verbose=translation_configs[
                                                                  'verbose'] if 'verbose' in translation_configs else False)
            translated_subs.append(translated_sub)
        return translated_subs

    @staticmethod
    def auto_sync(subs: SSAFile,
                  media_file: str,
                  **kwargs
                  ) -> SSAFile:
        """
        Uses (ffsubsync)[https://github.com/smacke/ffsubsync] to auto-sync subtitles to the media file

        :param subs: `SSAFile` file
        :param media_file: path of the media_file
        :param kwargs: configs to pass to ffsubsync (see :attr:`configs.ADVANCED_TOOLS_CONFIGS`)

        :return: `SSAFile` auto-synced
        """
        parser = make_parser()
        srtin_file = tempfile.NamedTemporaryFile(delete=False)
        srtout_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            srtin = srtin_file.name + '.ass'
            srtout = srtout_file.name + '.srt'
            subs.save(srtin)
            cmd = [media_file,
                   '-i', srtin,
                   '-o', srtout]
            for config_name in kwargs:
                value = kwargs[config_name]
                if value is None or value is False:
                    continue
                elif type(value) == bool and value is True:
                    cmd.append(f'--{config_name}')
                else:
                    cmd.append(f'--{config_name}')
                    cmd.append(f'{value}')
            parsed_args = parser.parse_args(cmd)
            retval = run(parsed_args)["retval"]
            synced_subs = pysubs2.load(srtout)
            return synced_subs
        finally:
            srtin_file.close()
            os.unlink(srtin_file.name)
            srtout_file.close()
            os.unlink(srtout_file.name)
    @staticmethod
    def merge_subs_with_video(subs: Dict[str, SSAFile],
                  media_file: str,
                  output_filename: str = None,
                  **kwargs
                  ) -> str:
        """
        Uses ffmpeg to merge subtitles into a video media file.
        You cna merge multiple subs at the same time providing a dict with (lang,`SSAFile` object) key,value pairs
        Example:
        ```python
            file = '../../assets/video/test1.webm'
            subs_ai = SubsAI()
            model = subs_ai.create_model('openai/whisper', {'model_type': 'tiny'})
            en_subs = subs_ai.transcribe(file, model)
            ar_subs = pysubs2.load('../../assets/video/test0-ar.srt')
            Tools.merge_subs_with_video2({'English': subs, "Arabic": subs2}, file)
        ```

        :param subs: dict with (lang,`SSAFile` object) key,value pairs
        :param media_file: path of the video media_file
        :param output_filename: Output file name (without the extension as it will be inferred from the media file)

        :return: Absolute path of the output file
        """
        metadata = ffmpeg.probe(media_file, select_streams="v")['streams'][0]
        assert metadata['codec_type'] == 'video', f'File {media_file} is not a video'


        srtin_files = {key: tempfile.NamedTemporaryFile(delete=False) for key in subs}
        try:
            in_file = pathlib.Path(media_file)
            if output_filename is not None:
                out_file = in_file.parent / f"{output_filename}{in_file.suffix}"
            else:
                out_file = in_file.parent / f"{in_file.stem}-subs-merged{in_file.suffix}"

            video = str(in_file.resolve())
            metadata_subs = {}
            ffmpeg_subs_inputs = []
            for i,lang in enumerate(srtin_files):
                srtin = srtin_files[lang].name + '.srt'
                subs[lang].save(srtin)
                ffmpeg_subs_inputs.append(ffmpeg.input(srtin)['s'])
                metadata_subs[f'metadata:s:s:{i}'] = "title=" + lang

            output_file = str(out_file.resolve())
            input_ffmpeg = ffmpeg.input(video)
            input_video = input_ffmpeg['v']
            input_audio = input_ffmpeg['a']
            output_ffmpeg = ffmpeg.output(
                input_video, input_audio, *ffmpeg_subs_inputs, output_file,
                vcodec='copy', acodec='copy',
                **metadata_subs
            )
            output_ffmpeg = ffmpeg.overwrite_output(output_ffmpeg)
            ffmpeg.run(output_ffmpeg)
        finally:
            for srtin_file in srtin_files.values():
                srtin_file.close()
                os.unlink(srtin_file.name)
        return str(out_file.resolve())

if __name__ == '__main__':
    file = '../../assets/video/test1.webm'
    subs_ai = SubsAI()
    model = subs_ai.create_model('openai/whisper', {'model_type': 'tiny'})
    subs = subs_ai.transcribe(file, model)
    subs.save('../../assets/video/test1.srt')
    subs2 = pysubs2.load('../../assets/video/test0-ar.srt')
    Tools.merge_subs_with_video2({'English': subs, "Arabic": subs2}, file)
    # subs.save('test1.srt')
