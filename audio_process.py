#!/usr/bin/python3

from os.path import splitext;
import numpy as np;
from pydub import AudioSegment;
from scipy.io import wavfile;
from librosa.beat import beat_track;

class AudioProcess(object):
  __opened = False;
  def __init__(self, audio_path = None):
    if audio_path is not None:
      self.load(audio_path);
  @property
  def data(self):
    return self.__data;
  @property
  def sample_width(self):
    return self.__sample_width;
  @property
  def channels(self):
    return self.__channels;
  @property
  def frame_rate(self):
    return self.__frame_rate;
  def load(self, audio_path):
    audiofile = AudioSegment.from_file(audio_path);
    # 1) data
    self.__data = np.reshape(np.array(audiofile.get_array_of_samples()), (-1, audiofile.channels)); # self.data.shape = (sample_num, channel_num)
    # 2) attributes
    self.__sample_width = audiofile.sample_width; # how many bytes for one sample
    self.__channels = audiofile.channels; # how many sound channels (whether it is a stereo audio)
    self.__frame_rate = audiofile.frame_rate; # how many samples per second
    # 3) flag to represent whether a file has been loaded
    self.__opened = True;
  def normalize(self):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    # return data in range [-1, 1]
    return self.__data / 2**(8*self.__sample_width - 1);
  def denormalize(self, data):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    return (data * 2**(8*self.__sample_width - 1)).astype(self.__data.dtype);
  def slice(self, start: int, length: int, normalized: bool = False):
    data = self.normalize() if normalized else self.__data;
    return data[start*self.__frame_rate:(start+length)*self.__frame_rate,:];
  def split(self, length: int, normalized: bool = False):
    data = self.normalize() if normalized else self.__data;
    segment_size = length * self.__frame_rate; # how many samples per slice
    return [data[x:x+segment_size,:] for x in np.arange(0, data.shape[0], segment_size)];
  def remove_silent_part(self, output: str = None):
    if output is None:
      output = "generated.wav";
    slices = self.split(1, True);
    energies = np.array([np.mean(slice**2, axis = 0) for slice in slices]); # energies.shape = (slice num, channel)
    thres = 0.5 * np.median(energies, axis = 0); # thres.shape = (channel,)
    index_of_segments_to_keep = np.where(np.logical_and.reduce(energies > thres, axis = 1));
    picked_slices = [self.denormalize(slices[i]) for i in index_of_segments_to_keep[0]];
    data = np.concatenate(picked_slices, axis = 0); # data.shape = (sample number, channel)
    wavfile.write(output, self.__frame_rate, data);
  def get_tempo(self,):
    for i in range(self.__data.shape[1]):
      tempo, beats = beat_track(self.__data[:,i].astype(np.float),  sr = self.__frame_rate, units="time");
      beats -= 0.05;
      

if __name__ == "__main__":

  ap = AudioProcess('samples/talk.mp3');
  print(ap.sample_width);
  print(ap.channels);
  print(ap.frame_rate);
  normalized = ap.normalize();
  sliced = ap.slice(2,2);
  splitted = ap.split(1000);
  ap.remove_silent_part();
  ap.load('samples/brahms_lullaby.mp3');
  ap.get_tempo();
