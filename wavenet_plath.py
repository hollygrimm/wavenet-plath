"""WaveNet Plath.
"""
"""
WaveNet Training code and utilities are licensed under APL2.0 from

Parag Mital
---------------------
https://github.com/pkmital/pycadl/blob/master/cadl/wavenet.py

Copyright 2017 Holly Grimm.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
import subprocess
from glob import glob
import numpy as np
import tensorflow as tf
from cadl import wavenet, vctk
from cadl import wavenet_utils as wnu
from cadl.utils import sample_categorical
from scipy.io import wavfile

def get_dataset(saveto='sounds', convert_mp3_to_16khzwav=False):
    """Convert MP3 files in 'saveto' directory to wav files.
    subfolders under the 'saveto' directory are considered chapters
    Each file name should be formatted CHAPTERNAME-UTTERANCE-DESCRIPTION.mp3
    ffmpeg must be installed to convert the files.
    Parameters
    ----------
    saveto : str, optional
        Directory to save the resulting dataset ['sounds']
    convert_to_16khz : bool, optional
        Description
    Returns
    -------
        dataset
    """
    if not os.path.exists(saveto):
        sys.exit("Error: '" + saveto + "' folder does not exist")

    wavs = glob('{}/**/*.16khz.wav'.format(saveto), recursive=True)
    if not wavs and convert_mp3_to_16khzwav:
        wavs = glob('{}/**/*.mp3'.format(saveto), recursive=True)
        for wav_i in wavs:
            subprocess.check_call(
                ['ffmpeg', '-i', wav_i, '-f', 'wav', '-ac', '1', '-ar', '16000', '-y', '%s.16khz.wav' % wav_i])

    wavs = glob('{}/**/*.16khz.wav'.format(saveto), recursive=True)

    if not wavs:
        sys.exit("Error: No 16khz wav files were found in '" + saveto + "'")        

    dataset = []
    for wav_i in wavs:
        chapter_i, utter_i = wav_i.split('/')[-2:]
        dataset.append({
            'name': wav_i,
            'chapter': chapter_i,
            'utterance': utter_i.split('-')[-2].strip('.wav')})
    return dataset

def train():
    """Train WaveNet on sound files
    Returns
    -------
        loss
    """    
    batch_size = 2
    filter_length = 2
    n_stages = 7
    n_layers_per_stage = 9
    n_hidden = 48
    n_skip = 384

    dataset = get_dataset(convert_mp3_to_16khzwav=True)
    it_i = 0
    n_epochs = 1000
    sequence_length = wavenet.get_sequence_length(n_stages, n_layers_per_stage)
    ckpt_path = 'plath-wavenet/wavenet_filterlen{}_batchsize{}_sequencelen{}_stages{}_layers{}_hidden{}_skips{}'.format(
        filter_length, batch_size, sequence_length, n_stages,
        n_layers_per_stage, n_hidden, n_skip)
    with tf.Graph().as_default(), tf.Session() as sess:
        net = wavenet.create_wavenet(
            batch_size=batch_size,
            filter_length=filter_length,
            n_hidden=n_hidden,
            n_skip=n_skip,
            n_stages=n_stages,
            n_layers_per_stage=n_layers_per_stage)
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        sess.run(init_op)
        if tf.train.latest_checkpoint(ckpt_path) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        batch = vctk.batch_generator
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(
                learning_rate=0.0002).minimize(net['loss'])
        var_list = [
            v for v in tf.global_variables() if v.name.startswith('optimizer')
        ]
        sess.run(tf.variables_initializer(var_list))
        writer = tf.summary.FileWriter(ckpt_path)
        for epoch_i in range(n_epochs):
            for batch_xs in batch(dataset, batch_size, sequence_length):
                loss, quantized, _ = sess.run(
                    [net['loss'], net['quantized'], opt],
                    feed_dict={net['X']: batch_xs})
                print(loss)
                if it_i % 100 == 0:
                    summary = sess.run(
                        net['summaries'], feed_dict={net['X']: batch_xs})
                    writer.add_summary(summary, it_i)
                    # save
                    saver.save(
                        sess,
                        os.path.join(ckpt_path, 'model.ckpt'),
                        global_step=it_i)
                it_i += 1
    return loss

def synthesize():
    """Synthesize 10 second wav files
    """
    batch_size = 2
    filter_length = 2
    n_stages = 7
    n_layers_per_stage = 9
    n_hidden = 48
    n_skip = 384
    total_length = 160000
    sequence_length = wavenet.get_sequence_length(n_stages, n_layers_per_stage)
    prime_length = sequence_length
    ckpt_path = 'plath-wavenet/wavenet_filterlen{}_batchsize{}_sequencelen{}_stages{}_layers{}_hidden{}_skips{}/'.format(
        filter_length, batch_size, sequence_length, n_stages,
        n_layers_per_stage, n_hidden, n_skip)

    dataset = get_dataset()
    batch = next(
        vctk.batch_generator(dataset, batch_size, prime_length))[0]

    sess = tf.Session()
    net = wavenet.create_wavenet(
        batch_size=batch_size,
        filter_length=filter_length,
        n_hidden=n_hidden,
        n_skip=n_skip,
        n_layers_per_stage=n_layers_per_stage,
        n_stages=n_stages,
        shift=False)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpt_path) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    else:
        print('Could not find checkpoint')

    synth = np.zeros([batch_size, total_length], dtype=np.float32)
    synth[:, :prime_length] = batch

    print('Synthesize...')
    for sample_i in range(0, total_length - prime_length):
        print('{}/{}/{}'.format(sample_i, prime_length, total_length), end='\r')
        probs = sess.run(
            net["probs"],
            feed_dict={net["X"]: synth[:, sample_i:sample_i + sequence_length]})
        idxs = sample_categorical(probs)
        idxs = idxs.reshape((batch_size, sequence_length))
        if sample_i == 0:
            audio = wnu.inv_mu_law_numpy(idxs - 128)
            synth[:, :prime_length] = audio
        else:
            audio = wnu.inv_mu_law_numpy(idxs[:, -1] - 128)
            synth[:, prime_length + sample_i] = audio

    for i in range(batch_size):
        wavfile.write('synthesis-{}.wav'.format(i), 16000, synth[i])


if __name__ == '__main__':
    train()