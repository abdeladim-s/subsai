#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API that the transcription models should follow
"""
from abc import ABC, abstractmethod
from pysubs2 import SSAFile


class AbstractModel(ABC):
    """
    Abstract Model class
    """
    def __init__(self, model_name=None, model_config={}):
        self.model_name = model_name
        self.model_config = model_config

    @abstractmethod
    def transcribe(self, media_file) -> SSAFile:
        """
        Transcribe the `media_file` to subtitles.

        example use case from pysubs2.whisper:

        .. code-block:: python
            :linenos:

        subs = SSAFile()
        for segment in segments:
            event = SSAEvent(start=make_time(s=segment["start"]), end=make_time(s=segment["end"]))
            event.plaintext = segment["text"].strip()
            subs.append(event)

        :param media_file: Path of the media file
        :return: Collection of SSAEvent(s) (see :mod:`pysubs2.ssaevent`)
        """
        pass
