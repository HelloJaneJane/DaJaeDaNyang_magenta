from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import collections
import io
import os

import tensorflow
import tensorflow.compat.v1 as tf
import magenta
import magenta.music as mm
from magenta.music import midi_synth
from magenta.music import constants

import pretty_midi
import bokeh
import bokeh.plotting
from IPython import display
import numpy as np
import pandas as pd
from scipy.io import wavfile
from six.moves import urllib
import tempfile

from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import pysynth as ps

def note_sequence_to_midi_file(sequence, output_file,
                               drop_events_n_seconds_after_last_note=None):
  """Convert NoteSequence to a MIDI file on disk.
  Time is stored in the NoteSequence in absolute values (seconds) as opposed to
  relative values (MIDI ticks). When the NoteSequence is translated back to
  MIDI the absolute time is retained. The tempo map is also recreated.
  Args:
    sequence: A NoteSequence.
    output_file: String path to MIDI file that will be written.
    drop_events_n_seconds_after_last_note: Events (e.g., time signature changes)
        that occur this many seconds after the last note will be dropped. If
        None, then no events will be dropped.
  """
  pretty_midi_object = note_sequence_to_pretty_midi(
      sequence, drop_events_n_seconds_after_last_note)
  with tempfile.NamedTemporaryFile() as temp_file:
    pretty_midi_object.write(temp_file)
    # Before copying the file, flush any contents
    temp_file.flush()
    # And back the file position to top (not need for Copy but for certainty)
    temp_file.seek(0)
    tf.gfile.Copy(temp_file.name, output_file, overwrite=True)

def note_sequence_to_pretty_midi(
    sequence, drop_events_n_seconds_after_last_note=None):
  """Convert NoteSequence to a PrettyMIDI.
  Time is stored in the NoteSequence in absolute values (seconds) as opposed to
  relative values (MIDI ticks). When the NoteSequence is translated back to
  PrettyMIDI the absolute time is retained. The tempo map is also recreated.
  Args:
    sequence: A NoteSequence.
    drop_events_n_seconds_after_last_note: Events (e.g., time signature changes)
        that occur this many seconds after the last note will be dropped. If
        None, then no events will be dropped.
  Returns:
    A pretty_midi.PrettyMIDI object or None if sequence could not be decoded.
  """
  ticks_per_quarter = sequence.ticks_per_quarter or constants.STANDARD_PPQ

  max_event_time = None
  if drop_events_n_seconds_after_last_note is not None:
    max_event_time = (max([n.end_time for n in sequence.notes] or [0]) +
                      drop_events_n_seconds_after_last_note)

  # Try to find a tempo at time zero. The list is not guaranteed to be in order.
  initial_seq_tempo = None
  for seq_tempo in sequence.tempos:
    if seq_tempo.time == 0:
      initial_seq_tempo = seq_tempo
      break

  kwargs = {}
  if initial_seq_tempo:
    kwargs['initial_tempo'] = initial_seq_tempo.qpm
  else:
    kwargs['initial_tempo'] = constants.DEFAULT_QUARTERS_PER_MINUTE

  pm = pretty_midi.PrettyMIDI(resolution=ticks_per_quarter, **kwargs)

  # Create an empty instrument to contain time and key signatures.
  instrument = pretty_midi.Instrument(0)
  pm.instruments.append(instrument)

  # Populate time signatures.
  for seq_ts in sequence.time_signatures:
    if max_event_time and seq_ts.time > max_event_time:
      continue
    time_signature = pretty_midi.containers.TimeSignature(
        seq_ts.numerator, seq_ts.denominator, seq_ts.time)
    pm.time_signature_changes.append(time_signature)

  # Populate key signatures.
  for seq_key in sequence.key_signatures:
    if max_event_time and seq_key.time > max_event_time:
      continue
    key_number = seq_key.key
    if seq_key.mode == seq_key.MINOR:
      key_number += _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET
    key_signature = pretty_midi.containers.KeySignature(
        key_number, seq_key.time)
    pm.key_signature_changes.append(key_signature)

  # Populate tempos.
  # TODO(douglaseck): Update this code if pretty_midi adds the ability to
  # write tempo.
  for seq_tempo in sequence.tempos:
    # Skip if this tempo was added in the PrettyMIDI constructor.
    if seq_tempo == initial_seq_tempo:
      continue
    if max_event_time and seq_tempo.time > max_event_time:
      continue
    tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
    tick = pm.time_to_tick(seq_tempo.time)
    # pylint: disable=protected-access
    pm._tick_scales.append((tick, tick_scale))
    pm._update_tick_to_time(0)
    # pylint: enable=protected-access

  # Populate instrument names by first creating an instrument map between
  # instrument index and name.
  # Then, going over this map in the instrument event for loop
  inst_infos = {}
  for inst_info in sequence.instrument_infos:
    inst_infos[inst_info.instrument] = inst_info.name

  # Populate instrument events by first gathering notes and other event types
  # in lists then write them sorted to the PrettyMidi object.
  instrument_events = collections.defaultdict(
      lambda: collections.defaultdict(list))
  for seq_note in sequence.notes:
    instrument_events[(seq_note.instrument, seq_note.program,
                       seq_note.is_drum)]['notes'].append(
                           pretty_midi.Note(
                               seq_note.velocity, seq_note.pitch,
                               seq_note.start_time, seq_note.end_time))
  for seq_bend in sequence.pitch_bends:
    if max_event_time and seq_bend.time > max_event_time:
      continue
    instrument_events[(seq_bend.instrument, seq_bend.program,
                       seq_bend.is_drum)]['bends'].append(
                           pretty_midi.PitchBend(seq_bend.bend, seq_bend.time))
  for seq_cc in sequence.control_changes:
    if max_event_time and seq_cc.time > max_event_time:
      continue
    instrument_events[(seq_cc.instrument, seq_cc.program,
                       seq_cc.is_drum)]['controls'].append(
                           pretty_midi.ControlChange(
                               seq_cc.control_number,
                               seq_cc.control_value, seq_cc.time))

  for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys()):
    # For instr_id 0 append to the instrument created above.
    if instr_id > 0:
      instrument = pretty_midi.Instrument(prog_id, is_drum)
      pm.instruments.append(instrument)
    else:
      instrument.is_drum = is_drum
    # propagate instrument name to the midi file
    instrument.program = prog_id
    if instr_id in inst_infos:
      instrument.name = inst_infos[instr_id]
    instrument.notes = instrument_events[
        (instr_id, prog_id, is_drum)]['notes']
    instrument.pitch_bends = instrument_events[
        (instr_id, prog_id, is_drum)]['bends']
    instrument.control_changes = instrument_events[
        (instr_id, prog_id, is_drum)]['controls']

  return pm

print(' * magenta version : ' + magenta.__version__)
print(' * tensorflow version : ' + tensorflow.__version__)


def melody_rnn(input_sequence):
  # Initialize the model.
  print("Initializing Melody RNN...")
  bundle = sequence_generator_bundle.read_bundle_file('/content/basic_rnn.mag')
  generator_map = melody_rnn_sequence_generator.get_generator_map()
  melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
  melody_rnn.initialize()

  # Model options. Change these to get different generated sequences! 

  input_sequence = twinkle_twinkle # change this to teapot if you want
  num_steps = 128 # change this for shorter or longer sequences
  temperature = 1.0 # the higher the temperature the more random the sequence.

  # Set the start time to begin on the next step after the last note ends.
  last_end_time = (max(n.end_time for n in input_sequence.notes)
                    if input_sequence.notes else 0)
  qpm = input_sequence.tempos[0].qpm 
  seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter
  total_seconds = num_steps * seconds_per_step

  generator_options = generator_pb2.GeneratorOptions()
  generator_options.args['temperature'].float_value = temperature
  generate_section = generator_options.generate_sections.add(
    start_time=last_end_time + seconds_per_step,
    end_time=total_seconds)

  # Ask the model to continue the sequence.
  return melody_rnn.generate(input_sequence, generator_options)

def music_vae_sample(model_id, model_config, num):
  music_vae = TrainedModel(
        configs.CONFIG_MAP[model_config], 
        batch_size=4, 
        checkpoint_dir_or_path=model_id+'.tar')

  generated_sequences = music_vae.sample(n=num, length=80, temperature=1.0)

  cnt=1
  for ns in generated_sequences:
    note_sequence_to_midi_file(ns,'vae_sample_'+model_id+'_%d.mid'%(cnt))
    cnt += 1

def music_vae_interpolate(sequence1, sequence2, model_id, model_config, num):
  music_vae = TrainedModel(
      configs.CONFIG_MAP[model_config], 
      batch_size=4, 
      checkpoint_dir_or_path=model_id+'.tar')

  note_sequences = music_vae.interpolate(
        sequence1, sequence2,
        num_steps=num,
        length=32)

  # Concatenate them into one long sequence, with the start and 
  # end sequences at each end. 
  return mm.sequences_lib.concatenate_sequences(note_sequences)


def get_simple_twinkle():
  twinkle_twinkle = music_pb2.NoteSequence()
  twinkle_twinkle.notes.add(pitch=60, start_time=0.0, end_time=0.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=1.0, end_time=1.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=1.5, end_time=2.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=69, start_time=2.0, end_time=2.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=69, start_time=2.5, end_time=3.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=3.0, end_time=4.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=65, start_time=4.0, end_time=4.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=65, start_time=4.5, end_time=5.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=64, start_time=5.0, end_time=5.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=64, start_time=5.5, end_time=6.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=62, start_time=6.0, end_time=6.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=62, start_time=6.5, end_time=7.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=60, start_time=7.0, end_time=8.0, velocity=80) 
  twinkle_twinkle.total_time = 8
  twinkle_twinkle.tempos.add(qpm=60)
  note_sequence_to_midi_file(twinkle_twinkle, 'twinkle_twinkle.mid')

def get_simple_teapot():
  teapot = music_pb2.NoteSequence()
  teapot.notes.add(pitch=69, start_time=0, end_time=0.5, velocity=80)
  teapot.notes.add(pitch=71, start_time=0.5, end_time=1, velocity=80)
  teapot.notes.add(pitch=73, start_time=1, end_time=1.5, velocity=80)
  teapot.notes.add(pitch=74, start_time=1.5, end_time=2, velocity=80)
  teapot.notes.add(pitch=76, start_time=2, end_time=2.5, velocity=80)
  teapot.notes.add(pitch=81, start_time=3, end_time=4, velocity=80)
  teapot.notes.add(pitch=78, start_time=4, end_time=5, velocity=80)
  teapot.notes.add(pitch=81, start_time=5, end_time=6, velocity=80)
  teapot.notes.add(pitch=76, start_time=6, end_time=8, velocity=80)
  teapot.total_time = 8
  teapot.tempos.add(qpm=60)
  note_sequence_to_midi_file(teapot, 'teapot.mid')

def get_simple_drum():
  drums = music_pb2.NoteSequence()
  drums.notes.add(pitch=36, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=38, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=46, start_time=0, end_time=0.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=0.25, end_time=0.375, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=0.375, end_time=0.5, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=50, start_time=0.5, end_time=0.625, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=36, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=38, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=45, start_time=0.75, end_time=0.875, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=36, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=46, start_time=1, end_time=1.125, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=42, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=48, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
  drums.notes.add(pitch=50, start_time=1.25, end_time=1.375, is_drum=True, instrument=10, velocity=80)
  drums.total_time = 1.375
  drums.tempos.add(qpm=60)
  note_sequence_to_midi_file(drums, 'drums.mid')



# # 기본 mid 파일 만들기
# get_simple_twinkle()
# get_simple_teapot()
# get_simple_drum()

# # melody RNN으로 기본 파일 뒤에 이어붙이기
# note_sequence_to_midi_file(melody_rnn(twinkle_twinkle), 'twinkle_melodyRNN.mid')
# note_sequence_to_midi_file(melody_rnn(teapot), 'teapot_melodyRNN.mid')

# # music VAE로 랜덤 작곡 파일 만들기
# music_vae_sample('cat-mel_2bar_big','cat-mel_2bar_big',4)
# music_vae_sample('hierdec-mel_16bar','hierdec-mel_16bar',6)
# music_vae_sample('hierdec-trio_16bar','hierdec-trio_16bar',6)
# music_vae_sample('cat-drums_2bar_small_hikl','cat-drums_2bar_small',4)
# music_vae_sample('cat-drums_2bar_small_lokl','cat-drums_2bar_small',4)
# music_vae_sample('groovae_2bar_humanize','groovae_2bar_humanize',4)

# # music VAE로 두 파일 interpolate
# note_sequence_to_midi_file(music_vae_interpolate(twinkle_twinkle, teapot, 'cat-mel_2bar_big','cat-mel_2bar_big',8), 'twinkle_teapot_interpol.mid')