#!/usr/bin/python3

from os.path import splitext;
import numpy as np;
from pydub import AudioSegment;

class AudioProcess(object):
  def __init__(self, audio_path):
    audiofile = AudioSegment.from_file(audio_path);
    channel_num = audiofile.channels;
    self.data = np.reshape(np.array(audiofile.get_array_of_samples()), (-1,channel_num)); # self.data.shape = (sample_num, channel_num)
    self.fs = audiofile.frame_rate;

if __name__ == "__main__":

  ap = AudioProcess('brahms_lullaby.mp3');
