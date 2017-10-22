# wavenet_experiment

Playing with a quick reimplementation of ibab's wavenet.  You may want to
have a look at the readme in [ibab's repo](https://github.com/ibab/tensorflow-wavenet)

- requirements:
  - Tensorflow >= 1.0
  - [librosa](https://github.com/librosa/librosa) for audio.

- Training:  wn_trainer.py
- Generation: synthesis.py

Training corpus:
   [CMU arctic](http://www.festvox.org/cmu_arctic/) (around 115Mb)

with CMU arctic slt and the default parameters, one epoch of training is about
11000 steps.  You will need to wait ~30 epochs or so before things start to
sound OK in generation.

Surprisingly, even though CMU artic is very small, I get better results training
with it than with the much bigger VCTK corpus.  I'm not sure why.

To run this:

```sh
$ ./wn_trainer.py -l {logdir} -o {checkpoint-file} -d {training_database}
```
The training database is a simple text file. It contains lines of the form
```
    {filename} user# : left_phone#1 right_phone#1 ... left_phone#N right_phone#N : log_f0#1 ... log_F0#N
```
the phone labels are at 100/second.  The log_f0 labels are the same rate.
I provide a sample database for slt via the file 'slt_trn.txt' and 'slt_tst.txt'.
You will probably have to change the path names in those files.

then after training a while:

```sh
$ ./synthesis.py -p {params file} -i {checkpoint-file-#####} -a {alignment file} -o {out.wav}
```
The alignment file is the same format as the training database.  The wav file
supplied in it is ignored during generation.

With the default parameters you can run on a 4Gb NVidia GTX-1050-ti.
With the defaults, it takes ~0.45 seconds per minibatch in training
on that card.  On a GTX 1080-ti it's about 0.13 seconds per minibatch.
CPU-only generation runs at about 150 samples per second on my Core i-5 6600K.
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

### Advantages over the ibab setup

- Uses phoneme and lf0 labels in conditional training.
- Simpler code. Generation is the same code as training.
- Batch normalization is an option.
- l2 regularization is an option
- dropout is an option.  Turning it on for an initial pass of training may be a
  good idea.
- either one-hot or scalar input features, both in generation and training.
  - scalar input produces lower cross-entropy, but 1-hot sounds better so far.
- You can train looking more than one sample into the future (last time I checked)
- I get better accuracy and faster convergence, at least last time I compared.
- You can use N-point à trous convolutions in generation.
- You can train a temporally-reversed model (probably needs to be updated.)
- Last I checked, ibab's setup changes the audio chunk size with different
  batches.  Doing this slows things down dramatically.  Because I carry
  context across chunks I can use constant chunk sizes without overlap.

There are of course some disadvantages too.

Here is an output sample: [sample_output.wav](http://github.com/cbquillen/wavenet_experiment/blob/biphone/sample_output.wav)

