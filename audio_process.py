#!/usr/bin/python3

from os.path import splitext;
import numpy as np;
from pydub import AudioSegment;
from scipy.io import wavfile;

class AudioProcess(object):
  def __init__(self, audio_path):
    if splitext(audio_path)[1] not in ['.wav', '.mp3']:
      raise Exception('currently only support wav and mp3 files!');
    if splitext(audio_path)[1] == '.wav':
      self.fs, self.data = wavfile.read(audio_path);
    elif splitext(audio_path)[1] == '.mp3':
      audiofile = AudioSegment.from_file(audio_path);
      self.data = np.array(audiofile.get_array_of_samples());
      self.fs = audiofile.frame_rate;

if __name__ == "__main__":

  ap = AudioProcess('brahms_lullaby.mp3');
