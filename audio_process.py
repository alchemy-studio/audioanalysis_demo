#!/usr/bin/python3

from os.path import splitext;
from typing import List;
import numpy as np;
from pydub import AudioSegment;
from scipy.io import wavfile;
from librosa import note_to_hz, hz_to_note, cqt, cqt_frequencies, stft, fft_frequencies, amplitude_to_db, display;
from librosa.beat import beat_track;
import pyaudio;
import struct;
import cv2;
import matplotlib.pyplot as plt;
import matplotlib
matplotlib.use('TkAgg')

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
  def normalize(self, data: np.array = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    if data is None: data = self.__data;
    # return data in range [-1, 1]
    return data / 2**(8*self.__sample_width - 1);
  def denormalize(self, data):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    return (data * 2**(8*self.__sample_width - 1)).astype(self.__data.dtype);
  def slice(self, start: int, length: int, normalized: bool = False, output: str = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    data = self.normalize() if normalized else self.__data;
    retval = data[start*self.__frame_rate:(start+length)*self.__frame_rate,:];
    if output is not None:
      assert splitext(output)[1] == '.wav';
      wavfile.write(output, self.__frame_rate, retval);
    return retval;
  def split(self, length: int, normalized: bool = False, output: str = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    data = self.normalize() if normalized else self.__data;
    segment_size = length * self.__frame_rate; # how many samples per slice
    retval = [data[x:x+segment_size,:] for x in np.arange(0, data.shape[0], segment_size)];
    if output is not None:
      assert splitext(output)[1] == '.wav';
      wavfile.write(output, self.__frame_rate, retval);
    return retval;
  def split_channels(self, output: str = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    channels = np.split(self.__data, self.__channels, -1); # channels = list[sample number x 1]
    if output is not None:
      assert splitext(output)[1] == '.wav';
      for i, channel in enumerate(channels):
        wavfile.write(splitext(output)[0] + str(i) + splitext(output)[1], channel, channel);
    return channels;
  def join_channels(self, channels: List[np.array], output: str = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    retval = np.concatenate(channels, axis = -1); # retval.shape = (sample number, channels)
    if output is not None:
      assert splitext(output)[1] == '.wav';
      wavfile.write(output, self.__frame_rate, retval);
    return retval;
  def remove_silent_part(self, output: str = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    if output is None:
      output = "generated.wav";
    slices = self.split(1, True);
    energies = np.array([np.mean(slice**2, axis = 0) for slice in slices]); # energies.shape = (slice num, channel)
    thres = 0.5 * np.median(energies, axis = 0); # thres.shape = (channel,)
    index_of_segments_to_keep = np.where(np.logical_and.reduce(energies > thres, axis = 1));
    picked_slices = [self.denormalize(slices[i]) for i in index_of_segments_to_keep[0]];
    data = np.concatenate(picked_slices, axis = 0); # data.shape = (sample number, channel)
    wavfile.write(output, self.__frame_rate, data);
  def get_tempo(self, just_beats = False):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    tempo_channels = list();
    # 1) create frames representing a beat which lasts for 0.2 second
    samples = np.arange(0, 0.2, 1 / self.__frame_rate); # how many frames for a beat
    amp_mod = 0.2 / (np.sqrt(samples) + 0.2) - 0.2; # amplitude decay, range in [-0.2, 0.8]
    amp_mod[amp_mod < 0] = 0; # filter sub-zero part, range in [0, 0.8]
    x = np.max(self.__data) * np.cos(2 * np.pi * samples * 220) * amp_mod; # generate samples with scaled amplitude
    # 2) generate audio frames containing beats which is as long as the loaded audio
    beat_channels = list();
    for i in range(self.__data.shape[1]):
      # detect beats for every single channel of the loaded audio
      # NOTE: beats is a list of time (seconds) which are picked as beats for tempo
      tempo, beats = beat_track(self.__data[:,i].astype(np.float32),  sr = self.__frame_rate, units="time");
      beat_channels.append(beats);
      #beats -= 0.05;
      tempo_channel = np.zeros_like(self.__data[:,i]); # temp_channel.shape = (sample number)
      for ib, b in enumerate(beats):
        sample_periods = np.arange(0, 0.2, 1 / self.__frame_rate);
        amp_mod = 0.2 / (np.sqrt(sample_periods) + 0.2) - 0.2; # amplitude decay, range in [-0.2, 0.8]
        amp_mod[amp_mod < 0] = 0; # filter sub-zero part, range in [0, 0.8]
        x = np.max(self.__data) * np.cos(2 * np.pi * sample_periods * 220) * amp_mod;
        tempo_channel[int(self.__frame_rate * b): int(self.__frame_rate * b) + int(x.shape[0])] = x.astype(np.int16);
      tempo_channels.append(np.expand_dims(tempo_channel, axis = -1));
    return tempo_channels if just_beats == False else beat_channels;
  def from_microphone(self, sample_window: float = 0.2, frame_rate: int = 8000, channels: int = 1, count: int = -1):
    # sample_window: how long (second) each sample segment is
    pa = pyaudio.PyAudio();
    stream = pa.open(format = pyaudio.paInt16, channels = channels, rate = frame_rate, input = True, frames_per_buffer = int(frame_rate * sample_window));
    i = 0;
    while True if count < 0 else i < count:
      block = stream.read(int(frame_rate * sample_window)); # get samples from microphone
      shorts = struct.unpack("%dh" % (len(block) / 2), block);
      data = np.array(list(short)).astype(np.int16);
      i += 1;
    # TODO: save audio from microphone to self
  def cqt(self, data: np.array = None, hop_lengths: List[int] = None, bins_per_octave: int = 12):
    # data: shape = (sample number, channel number)
    # hop_length: how many samples are between two selected sample segments
    if self.__opened == False:
      raise Exception('load an audio file first!');
    if hop_lengths is None:
      hop_lengths = [512] * (self.__channels if data is None else data.shape[-1]);
    assert len(hop_lengths) == self.__channels if data is None else len(hop_lengths) == data.shape[-1];
    normalized = self.normalize(data);
    channels = list();
    for i in range(normalized.shape[-1]):
      normalized_channel = normalized[:,i];
      channel_results = cqt(normalized_channel, self.__frame_rate, hop_lengths[i], fmin = note_to_hz('A0'), n_bins = 88, bins_per_octave = bins_per_octave); # results.shape = (84, hop number)
      channels.append(channel_results);
    spectrum = np.stack(channels, axis = 0); # spectrum.shape = (channel number, 88, hop number)
    freqs = cqt_frequencies(88, fmin = note_to_hz('A0'), bins_per_octave = bins_per_octave);
    return spectrum, freqs;
  def stft(self, data: np.array = None, hop_lengths: List[int] = None):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    if hop_lengths is None:
      hop_lengths = [512] * (self.__channels if data is None else data.shape[-1]);
    assert len(hop_lengths) == self.__channels if data is None else len(hop_lengths) == data.shape[-1];
    normalized = self.normalize(data);
    channels = list();
    for i in range(normalized.shape[-1]):
      normalized_channel = normalized[:,i];
      channel_results = stft(normalized_channel, 22050, hop_lengths[i]);
      channels.append(channel_results); # channel_results.shape = (1 + 22050/2, hop number)
    spectrum = np.stack(channels, axis = 0); # spectrum.shape = (channel number, 1 + 22050/2, hop number)
    freqs = fft_frequencies(self.__frame_rate, 22050); # freqs.shape = (1 + 22050/2)
    return spectrum, freqs;
  def note_threshold_scaled_by_RMS(self, buffer_rms):
    note_threshold = 1000.0 * (4 / 0.090) * buffer_rms
    return note_threshold
  def pitch_spectral_hps(self, X, freq_buckets, f_s, buffer_rms):

    """
    NOTE: This function is from the book Audio Content Analysis repository
    https://www.audiocontentanalysis.org/code/pitch-tracking/hps-2/
    The license is MIT Open Source License.
    And I have modified it. Go to the link to see the original.
    computes the maximum of the Harmonic Product Spectrum
    Args:
        X: spectrogram (dimension FFTLength X Observations)
        f_s: sample rate of audio data
    Returns:
        f HPS maximum location (in Hz)
    """

    # initialize
    iOrder = 4
    f_min = 65.41   # C2      300
    # f = np.zeros(X.shape[1])
    f = np.zeros(len(X))

    iLen = int((X.shape[0] - 1) / iOrder)
    afHps = X[np.arange(0, iLen)]
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] - 1)))

    # compute the HPS
    for j in range(1, iOrder):
        X_d = X[::(j + 1)]
        afHps *= X_d[np.arange(0, iLen)]

    ## Uncomment to show the original algorithm for a single frequency or note. 
    # f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)
    ## find max index and convert to Hz
    # freq_out = (f + k_min) / (X.shape[0] - 1) * f_s / 2

    note_threshold = self.note_threshold_scaled_by_RMS(buffer_rms)

    all_freq = np.argwhere(afHps[np.arange(k_min, afHps.shape[0])] > note_threshold)
    # find max index and convert to Hz
    freqs_out = (all_freq + k_min) / (X.shape[0] - 1) * f_s / 2

    
    x = afHps[np.arange(k_min, afHps.shape[0])]
    freq_indexes_out = np.where( x > note_threshold)
    freq_values_out = x[freq_indexes_out]

    # print("\n##### x: " + str(x))
    # print("\n##### freq_values_out: " + str(freq_values_out))

    max_value = np.max(afHps[np.arange(k_min, afHps.shape[0])])
    max_index = np.argmax(afHps[np.arange(k_min, afHps.shape[0])])

    # Turns 2 level list into a one level list.
    freqs_out_tmp = []
    for freq, value  in zip(freqs_out, freq_values_out):
        freqs_out_tmp.append((freq[0], value))
    
    return freqs_out_tmp
  def scale_recognition(self,):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    beat_channels = self.get_tempo(just_beats = True);
    channels = list();
    for channel, beats in enumerate(beat_channels):
      print("processing channel %d" % channel);
      channels.append(list());
      for i in range(len(beats)-1):
        print('processing %d/%d' % (i, len(beats)-1));
        segment = self.__data[int(beats[i]*self.__frame_rate):int(beats[i+1]*self.__frame_rate),channel:channel+1]; # segment.shape = (sample number, channel number = 1)
        #segment = segment[int(segment.shape[0]/4):int(segment.shape[0]*4/4),:];
        hop_length = int(2 ** np.floor(np.log2(segment.shape[0])));
        spectrum, freqs = self.stft(segment, [hop_length]); # spectrum.shape = (channel_number = 1, 1 + 22050/2, hop number <= 2)
        spectrum = np.abs(spectrum[0,:,0]); # spectrum.shape = (1 + 22050/2)
        # remove dc offset
        spectrum[0:3] = np.zeros_like(spectrum[0:3]);
        rms = np.sqrt(np.mean(segment ** 2));
        detected_freqs = self.pitch_spectral_hps(spectrum, freqs, self.__frame_rate, rms);
        detected_notes = [hz_to_note(freq[0]) for freq in detected_freqs if note_to_hz('A0') <= freq[0] <= note_to_hz('C8')]; # detected_notes.shape = (note number)
        channels[-1].append(detected_notes);
    return channels;
  def visualize(self, channel = 0, output = 'visualize.avi'):
    if self.__opened == False:
      raise Exception('load an audio file first!');
    assert channel < self.__channels;
    beat_channels = self.get_tempo(just_beats = True);
    period = beat_channels[channel][1] - beat_channels[channel][0];
    fps = 1/period;
    writer = None;
    for i in range(len(beat_channels[channel])-1):
      print('processing %d/%d' % (i, len(beat_channels[channel])-1));
      segment = self.__data[int(beat_channels[channel][i] * self.__frame_rate):int(beat_channels[channel][i+1]*self.__frame_rate),channel:channel+1];
      hop_length = int(2 ** np.floor(np.log2(segment.shape[0])));
      spectrum, freqs = self.cqt(segment); # spectrum.shape = (channel number = 1, 88, hop number <= 2)
      CQT = amplitude_to_db(spectrum[0], ref = np.max);
      fig = plt.figure(figsize = (12,8));
      display.specshow(CQT, x_axis = 'time', y_axis = 'cqt_hz');
      plt.colorbar(format = '%+2.0f dB');
      plt.title('Constant-Q power spectrogram (Hz)');
      fig.canvas.draw();
      image = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep='');
      image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,));
      #'''
      cv2.imshow('', image);
      cv2.waitKey(int(period));
      #'''
      if writer is None:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'XVID'), fps, fig.canvas.get_width_height()[::-1]);
      writer.write(image);
    writer.release();

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
  channels = ap.split_channels();
  tempo_channels = ap.get_tempo();
  for i,(c,t) in enumerate(zip(channels, tempo_channels)):
    ap.join_channels([c,t], str(i) + ".wav");
  #ap.from_microphone(count = 10);
  
  channels = ap.scale_recognition();
  with open('notes.txt','w') as f:
    for notes in channels[0]:
      line = ','.join(notes);
      f.write(line + "\n");
  
  #ap.visualize();
