#!/usr/bin/python3

from typing import List;
from os import remove;
from os.path import splitext;
import subprocess;
from tempfile import NamedTemporaryFile;

class AudioProcess(object):
  typical_sampling_frequencies = [8000, 16000,44100];

  def __init__(self,):
    pass

  def get_audio_from_video(self, video_path: str, output: str = None)->None:
    if output is None:
      output = splitext(video_path)[0] + '.wav';
    subprocess.run(['ffmpeg', '-i', video_path, output]);

  def resample_audio(self, audio_path: str, frequency: int = 8000, channels: int = 2, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_" + str(frequency) + "_" + ('mono' if channels == 1 else 'stereo') + ".wav";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ar', frequency, '-ac', channels, output]);

  def show_audio_attributes(self, audio_path: str)->None:
    subprocess.run(['ffmpeg', '-i', audio_path]);

  def trim_audio(self, audio_path: str, start: int, length: int, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_trimmed.wav";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ss', start, '-t', length, output]);

  def concat_audio(self, audio_paths: List[str], output: str)->None:
    if output is None:
      output = 'concated.wav';
    f = NamedTemporaryFile();
    list_file = f.name;
    for path in audio_path:
      list_file.write('file ' + path + '\n');
    f.close();
    subprocess.run(['ffmpeg', '-f', 'concat', '-i', list_file, '-c', 'copy', output]);
    remove(list_file);


