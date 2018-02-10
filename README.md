# wavenet_experiment

Playing with a quick reimplementation of ibab's wavenet.  You may want to
have a look at the readme in [ibab's repo](https://github.com/ibab/tensorflow-wavenet)

This version uses a single asymmetric Laplace distribution to model the output distribution.

- requirements:
  - Tensorflow >= 1.0
  - [librosa](https://github.com/librosa/librosa) for audio.

- Training:  wn_trainer.py
- Generation: synthesis.py

Training corpora:
   [CMU arctic](http://www.festvox.org/cmu_arctic/) (around 115Mb)
   [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (around 10.4GB, [Alternative host](http://www.udialogue.org/download/cstr-vctk-corpus.html))


with CMU arctic slt and the default parameters, one epoch of training is about
11000 steps.  You will need to wait ~30 epochs or so before things start to
sound OK in generation.

The VCTK corpus can also be used for training.  I get reasonable results training on a 10-speaker subset of the corpus, for which I supply phoneme labels.

To run this:

```sh
$ ./wn_trainer.py -l {logdir} -o {checkpoint-file} -d {training_database} -a {audio_root_dir}
```
The training database is a simple text file. It contains lines of the form
```
    {filename} user# : context#1_phone#1 .. context#N_phone#1 ... context#1_phone#M ... context#N_phone#M : log_f0#1 ... log_F0#N
```
The {filename} is the audio .wav file.  It is specified relative to {audio_root_dir}.

the phone labels are at 100/second.  The log_f0 labels are the same rate.
I provide a sample database for slt via the files 'slt_trn.txt' and 'slt_tst.txt'.
I also provide a sample database for a subset of VCTK via vctk_low_subset.txt.

then after training a while:

```sh
$ ./synthesis.py -p {params file} -i {checkpoint-file-#####} -a {alignment file} -o {out.wav}
```
The alignment file is the same format as the training database.  The wav file
supplied in it is ignored during generation.

With the default parameters you can run on a 4Gb NVidia GTX-1050-ti.
With the defaults, it takes ~0.45 seconds per minibatch in training
on that card.  On a GTX 1080-ti it's about 0.13 seconds per minibatch.
CPU-only generation runs at about 125 samples per second on my Core i-5 6600K.
You will probably want to train a minimum of 400000 steps, which will take
about 14 hours on the GTX 1080-ti.

If you have a card with smaller memory, you can run with a smaller
audio chunk size or fewer chunks per minibatch.  I carry context correctly
across chunks, so there should be no impact on modeling accuracy.  It
does change the effective batch-size used for derivatives, so you may
need to adjust the learning rate.

Many parameters must be changed using a parameter file.  Use "-p *params_file*".
I include an example *params.txt* file.  It is executable Python.  You will
need to suppy a parameter file for generation.  Generally you should
use the same one that you used in training.

Here is an output sample: [sample_output.wav](http://github.com/cbquillen/wavenet_experiment/blob/2-sided-laplace/sample_output.wav)

