#!/usr/bin/python3

from enum import Enum;
from typing import List;
from os import remove;
from os.path import splitext;
import subprocess;
from tempfile import NamedTemporaryFile;

class Channel(Enum):
  Left = 1;
  Right = 2;

class FFMPEG(object):
  typical_sampling_frequencies = [8000, 16000,44100];

  def __init__(self,):
    pass

  @staticmethod
  def from_video(video_path: str, output: str = None)->None:
    if output is None:
      output = splitext(video_path)[0] + '.flac';
    subprocess.run(['ffmpeg', '-i', video_path, output]);

  @staticmethod
  def resample(audio_path: str, frequency: int = 8000, channels: int = 2, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_" + str(frequency) + "_" + ('mono' if channels == 1 else 'stereo') + ".flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ar', str(frequency), '-ac', str(channels), output]);

  @staticmethod
  def show_attributes(audio_path: str)->None:
    subprocess.run(['ffmpeg', '-i', audio_path]);

  @staticmethod
  def slice(audio_path: str, start: int, length: int, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_sliced.flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-ss', str(start), '-t', str(length), output]);

  @staticmethod
  def concat(audio_paths: List[str], output: str = None)->None:
    if output is None:
      output = 'concated.flac';
    f = NamedTemporaryFile();
    list_file = f.name;
    for path in audio_paths:
      list_file.write('file ' + path + '\n');
    f.close();
    subprocess.run(['ffmpeg', '-f', 'concat', '-i', list_file, '-c', 'copy', output]);
    remove(list_file);

  @staticmethod
  def split(audio_path: str, length: int = 1, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_splitted%05d.flac';
    subprocess.run(['ffmpeg', '-i', audio_path, '-f', 'segment', '-segment_time', str(length), '-c', 'copy', output]);

  @staticmethod
  def switch_channels(audio_path: str, output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_switched.flac';
    subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.1', '-map_channel', '0.0.0', output]);

  @staticmethod
  def join_channels(audio_paths: List[str], output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + '_joined.flac';
    inputs = list();
    for path in audio_paths:
      inputs.append('-i');
      inputs.append(path);
    subprocess.run(['ffmpeg', *inputs, "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]", "-map", "[a]", output]);

  @staticmethod
  def split_channels(audio_path: str, left_output: str = None, right_output: str = None)->None:
    if left_output is None:
      left_output = splitext(audio_path)[0] + "_left.flac";
    if right_output is None:
      right_output = splitext(audio_path)[0] + "_right.flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.0', left_output, '-map_channel', '0.0.1', right_output]);

  @staticmethod
  def mute_channel(audio_path: str, channel: Channel = Channel.Left, output: str = None)->None:
    assert channel in ['left', 'right'];
    if output is None:
      output = splitext(audio_path)[0] + "_muted.flac";
    if channel == Channel.Left:
      subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '-1', '-map_channel', '0.0.1', output]);
    else:
      subprocess.run(['ffmpeg', '-i', audio_path, '-map_channel', '0.0.0', output, '-map_channel', '-1']);

  @staticmethod
  def volume_adjust(audio_path: str, rate: float = 1., output: str = None)->None:
    if output is None:
      output = splitext(audio_path)[0] + "_adjusted.flac";
    subprocess.run(['ffmpeg', '-i', audio_path, '-filter:a', "volume=" + str(round(rate,1)), output ]);

if __name__ == "__main__":

  FFMPEG.show_attributes('brahms_lullaby.mp3');
  FFMPEG.resample('brahms_lullaby.mp3');
  FFMPEG.slice('brahms_lullaby.mp3', 10, 5);
  FFMPEG.concat(['brahms_lullaby.mp3','brahms_lullaby.mp3']);
  FFMPEG.split('brahms_lullaby.mp3',10);
  FFMPEG.switch_channels('brahms_lullaby.mp3');
  FFMPEG.split_channels('brahms_lullaby.mp3');
  FFMPEG.join_channels(['brahms_lullaby_left.flac','brahms_lullaby_right.flac']);
  FFMPEG.mute_channel('brahms_lullaby.mp3', Channel.Left);
  FFMPEG.volume_adjust('brahms_lullaby.mp3', 0.5);
