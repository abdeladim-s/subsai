#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract SST AI models,
Every SST model should follow this API

Longer description of this module.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
from abc import ABC, abstractmethod
from pysubs2 import SSAFile


class AbstractModel(ABC):

    def __init__(self, model_name=None, model_config={}):
        self.model_name = model_name
        self.model_config = model_config

    @abstractmethod
    def transcribe(self, media_file) -> SSAFile:
        """
        Transcribe the `media_file` to subtitles.
        example use case from pysubs2.whisper
        subs = SSAFile()
        for segment in segments:
            event = SSAEvent(start=make_time(s=segment["start"]), end=make_time(s=segment["end"]))
            event.plaintext = segment["text"].strip()
            subs.append(event)
        :return: Collection of SSAEvent(s) (see `pysubs2.ssaevent`)
        """
        pass

