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

  def from_video(self, video_path: str, output: str = None)->None:
    if output is None:
      output = splitext(video_path)[0] + '.flac';
    subprocess.run(['ffmpeg', '-i', video_path, output]);

  def resample(self, audio_path: str, frequency: int = 8000, channels: int = 2, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_" + str(frequency) + "_" + ('mono' if channels == 1 else 'stereo') + ".flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ar', frequency, '-ac', channels, output]);

  def show_attributes(self, audio_path: str)->None:
    subprocess.run(['ffmpeg', '-i', audio_path]);

  def slice(self, audio_path: str, start: int, length: int, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_sliced.flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ss', start, '-t', length, output]);

  def concat(self, audio_paths: List[str], output: str = None)->None:
    if output is None:
      output = 'concated.flac';
    f = NamedTemporaryFile();
    list_file = f.name;
    for path in audio_path:
      list_file.write('file ' + path + '\n');
    f.close();
    subprocess.run(['ffmpeg', '-f', 'concat', '-i', list_file, '-c', 'copy', output]);
    remove(list_file);

  def split(self, audio_path: str, length: int = 1, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_splitted%05d.flac';
    subprocess.run(['ffmpeg', '-i', audio_path, '-f', 'segment', '-segment_time', length, '-c', 'copy', output]);

  def switch_channels(self, audio_path: str, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_switched.flac';
    subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.1', '-map_channel', '0.0.0', output]);

  def join_channels(self, audio_paths: List[str], output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_joined.flac';
    inputs = list();
    for path in audio_paths:
      inputs.append('-i');
      inputs.append(path);
    subprocess.run(['ffmpeg', *inputs, "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]", "-map", "[a]", output]);

  def split_channels(self, audio_path: str, left_output: str = None, right_output: str = None)->None:
    if left_output is None:
      left_output = splitext(audio_path)[0] + "_left.flac";
    if right_output is None:
      right_output = splitext(audio_path)[0] + "_right.flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.0', left_output, '-map_channel', '0.0.1', right_output]);

  def mute_channels(self, audio_path: str, channel: str = 'left', output: str = None)->None:
    assert channel in ['left', 'right'];
    if output is None:
      output = splitext(audio_path)[0] + "_muted.flac";
    if channel == 'left':
      subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '-1', '-map_channel', '0.0.1', output]);
    else:
      subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.0', output, '-map_channel', '-1']);


