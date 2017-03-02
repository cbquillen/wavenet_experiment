# wavenet_experiment

Playing with a quick reimplementation of ibab's wavenet.  You may want to
have a look at the readme in [ibab's repo](https://github.com/ibab/tensorflow-wavenet)

- requirements:
  - Tensorflow 1.0
  - [librosa](https://github.com/librosa/librosa) for audio.

- Training:  wn_trainer.py
- Generation: wn_generate.py

Training corpus:
   [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (around 10.4GB, [Alternative host](http://www.udialogue.org/download/cstr-vctk-corpus.html))

with VCTK and the default parameters, one epoch of training is about 34000
steps.  You will need to wait an epoch or two before you can expect to see
anything reasonable in generation.

To run this:

```sh
$ ./wn_trainer.py -l {logdir} -o {checkpoint-file}
```

then after training a while:

```sh
$ ./wn_generate.py -p {params file} -i {checkpoint-file-#####} -n {n-samples} -o {out.wav}
```
The default parameters work for me on an 8Gb NVidia GTX-1070.
With these defaults, it takes ~1.0 seconds per minibatch in training
when using an audio chunk of 50000 samples (half of ibab's default.)
CPU-only generation runs at about 150 samples per second on my
Core i-5 6600K.  You will probably want to train a minimum of 100000
steps, which will take at least a day unless you have a better GPU card.

If you have a card with smaller memory, you can run with a smaller
audio chunk size.  I carry context correctly across chunks, so there
should be no impact on modeling accuracy.  It does change the effective
batch-size used for derivatives, so you may need to adjust the
learning rate.

Many parameters must be changed using a parameter file.  Use "-p *params_file*".
I include an example *params.txt* file.  It is executable Python.  You will
need to suppy a parameter file for generation.  Generally you should
use the same one that you used in training.

### Advantages over the ibab setup

- Simpler code. Generation is the same code as training.
- Batch normalization is an option.
- either one-hot or scalar input features, both in generation and training.
  - scalar input seems to produce better results
- You can train looking more than one sample into the future.
- I get better accuracy and faster convergence, at least last time I compared.
- You can use N-point à trous convolutions in generation.
- Last I checked, ibab's setup changes the audio chunk size with different
  batches.  Doing this slows things down dramatically.  Because I carry
  context across chunks I can use constant chunk sizes without overlap.

There are of course some disadvantages too.

Here is an output sample: [sample_output.wav](http://github.com/cbquillen/wavenet_experiment/blob/master/sample_output.wav)

