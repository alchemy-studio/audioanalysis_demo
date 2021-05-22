#!/usr/bin/python3

from os.path import splitext;
import numpy as np;
from pydub import AudioSegment;

class AudioProcess(object):
  __opened = False;
  def __init__(self, audio_path = None):
    if audio_path is not None:
      self.load(audio_path);
  def load(self, audio_path):
    audiofile = AudioSegment.from_file(audio_path);
    # 1) data
    self.__data = np.reshape(np.array(audiofile.get_array_of_samples()), (-1, audiofile.channels)); # self.data.shape = (sample_num, channel_num)
    # 2) attributes
    self.__sample_width = audiofile.sample_width;
    self.__channels = audiofile.channels;
    self.__frame_rate = audiofile.frame_rate; # how many samples per second
    # 3) flag to represent whether a file has loaded
    self.__opened = True;
  def normalize(self):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    # return data in range [-1, 1]
    return self.__data / 2**(8*self.__sample_width - 1);
  def slice(self, start: int, length: int, normalized: bool = False):
    data = self.normalize() if normalized else self.__data;
    return data[start*self.__frame_rate:(start+length)*self.__frame_rate,:];
  

if __name__ == "__main__":

  ap = AudioProcess('brahms_lullaby.mp3');
  normalized = ap.normalize();
  sliced = ap.slice(2,2);
