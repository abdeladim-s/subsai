#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SubsAI Command Line Interface (cli)
"""
import argparse
import importlib.metadata
from typing import List

__author__ = "abdeladim-s"
__contact__ = "https://github.com/abdeladim-s"
__copyright__ = "Copyright 2023,"
__license__ = "GPLv3"
__version__ = importlib.metadata.version('subsai')

__header__ = f"""
███████╗██╗   ██╗██████╗ ███████╗     █████╗ ██╗
██╔════╝██║   ██║██╔══██╗██╔════╝    ██╔══██╗██║
███████╗██║   ██║██████╔╝███████╗    ███████║██║
╚════██║██║   ██║██╔══██╗╚════██║    ██╔══██║██║
███████║╚██████╔╝██████╔╝███████║    ██║  ██║██║
╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝
                                            
Subs AI: Subtitles generation tool powered by OpenAI's Whisper and its variants.
Version: {__version__}               
===================================
"""

import json
import os
import pathlib

from subsai import SubsAI, Tools
from subsai.utils import available_translation_models, available_subs_formats

subs_ai = SubsAI()
tools = Tools()


def _handle_media_file(media_file_arg: List[str]):
    res = []
    for file in media_file_arg:
        if file.endswith('.txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line == '':
                        continue
                    res.append(pathlib.Path(line.strip()).resolve())
        else:
            res.append(pathlib.Path(file).resolve())
    return res


def _handle_configs(model_configs_arg: str):
    if model_configs_arg.endswith('.json'):
        with open(model_configs_arg, 'r') as file:
            data = file.read()
            return json.loads(data)
    return json.loads(model_configs_arg)


def run(media_file_arg: List[str],
        model_name,
        model_configs,
        destination_folder,
        subs_format,
        translation_model,
        translation_configs,
        translation_source_lang,
        translation_target_lang,
        ):
    files = _handle_media_file(media_file_arg)
    model_configs = _handle_configs(model_configs)
    print(f"[-] Model name: {model_name}")
    print(f"[-] Model configs: {'defaults' if model_configs == {} else model_configs}")
    print(f"---")
    print(f"[+] Initializing the model")
    model = subs_ai.create_model(model_name, model_configs)
    tr_model = None
    for file in files:
        print(f"[+] Processing file: {file}")
        if not file.exists():
            print(f"[*] Error: {file} does not exist -> continue")
            continue
        subs = subs_ai.transcribe(file, model)
        if destination_folder is not None:
            folder = pathlib.Path(destination_folder).absolute()
            if not folder.exists():
                print(f"[+] Creating folder: {folder}")
                os.makedirs(folder, exist_ok=True)
            file_name = folder / (file.stem + '.' + subs_format)
        else:
            file_name = file.parent / (file.stem + '.' + subs_format)
        if translation_model is not None:
            if tr_model is None:
                print(f"[+] Creating translation model: {translation_model}")
                tr_model = tools.create_translation_model(translation_model)
            print(f"[+] Translating from: {translation_source_lang} to {translation_target_lang}")
            translation_configs = _handle_configs(translation_configs)
            subs = tools.translate(subs=subs,
                                   source_language=translation_source_lang,
                                   target_language=translation_target_lang,
                                   model=tr_model,
                                   translation_configs=translation_configs)
        print(f"[+] Subtitles file saved to: {file_name}")
        subs.save(file_name)
    print('DONE!')


def main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('media_file', type=str, nargs='+', help="The path of the media file, a list of files, or a "
                                                                "text file containing paths for batch processing.")

    # Optional args
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('-m', '--model', default=SubsAI.available_models()[0],
                        help=f'The transcription AI models. Available models: {SubsAI.available_models()}')
    parser.add_argument('-mc', '--model-configs', default="{}",
                        help="JSON configuration (path to a json file or a direct "
                             "string)")
    parser.add_argument('-f', '--format', '--subtitles-format', default='srt',
                        help=f"Output subtitles format, available "
                             f"formats {available_subs_formats(include_extensions=False)}")
    parser.add_argument('-df', '--destination-folder', default=None,
                        help='The directory where the subtitles will be stored, default to the same folder where '
                             'the media file(s) is stored.')
    parser.add_argument('-tm', '--translation-model', default=None,
                        help=f"Translate subtitles using AI models, available "
                             f"models: {available_translation_models()}", )
    parser.add_argument('-tsl', '--translation-source-lang', default=None, help="Source language of the subtitles")
    parser.add_argument('-ttl', '--translation-target-lang', default=None, help="Target language of the subtitles")
    parser.add_argument('-tc', '--translation-configs', default="{}",
                        help="JSON configuration (path to a json file or a direct "
                             "string)")

    args = parser.parse_args()

    run(media_file_arg=args.media_file,
        model_name=args.model,
        model_configs=args.model_configs,
        destination_folder=args.destination_folder,
        subs_format=args.format,
        translation_model=args.translation_model,
        translation_configs=args.translation_configs,
        translation_source_lang=args.translation_source_lang,
        translation_target_lang=args.translation_target_lang)


if __name__ == '__main__':
    main()
