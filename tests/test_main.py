#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test file the main module

"""
import pathlib
from unittest import TestCase

import pysubs2
from pysubs2 import SSAFile

from subsai import SubsAI, Tools


class TestSubsAI(TestCase):
    subs_ai = SubsAI()
    model_name = 'openai/whisper'
    files = ['../assets/video/test1.mp4', '../assets/audio/test1.mp3']

    def test_available_models(self):
        available_models = self.subs_ai.available_models()
        self.assertIsInstance(available_models, list, 'This should return a list')

    def test_model_info(self):
        for model in self.subs_ai.available_models():
            info = self.subs_ai.model_info(model)
            self.assertIn('description', info, f'no description for model {model}')

    def test_config_schema(self):
        for model in self.subs_ai.available_models():
            schema = self.subs_ai.config_schema(model)
            self.assertIsInstance(schema, dict, 'the config schema needs to be a dict object')
            for config in schema:
                for field in ['type', 'description', 'options', 'default']:
                    self.assertIn(field, schema[config], f'No {field} in the schema of the {model} model')

    def test_create_model(self):
        for model in self.subs_ai.available_models():
            model_instance = self.subs_ai.create_model(model)
            self.assertIsNotNone(model_instance, f'Model {model} should not be None')

    def test_transcribe(self):
        for model in self.subs_ai.available_models():
            model_instance = self.subs_ai.create_model(model)
            for file in self.files:
                subs = model_instance.transcribe(file, model_instance)
                self.assertIsInstance(subs, SSAFile, 'transcribe function should return `pysubs2.SSAFile`')


class TestTools(TestCase):
    tools = Tools()
    file = '../assets/video/test1.webm'
    subs_file = '../assets/video/test1.srt'
    subs = pysubs2.load(subs_file)

    def test_translate(self):
        translated_subs = self.tools.translate(self.subs, 'English', 'Arabic')
        self.assertIsInstance(translated_subs, SSAFile)
        self.assertEqual(len(self.subs), len(translated_subs), 'Translation subs should have the same length as the '
                                                               'original subs')

    def test_auto_sync(self):
        synced_subs = self.tools.auto_sync(self.subs, self.file)
        self.assertIsInstance(synced_subs, SSAFile)


    def test_merge_subs_with_video(self):
        Tools.merge_subs_with_video2({'English': self.subs}, self.file, 'subs-merged')
        in_file = pathlib.Path(self.file)
        self.assertTrue((in_file.parent / f"subs-merged{in_file.suffix}").exists())
