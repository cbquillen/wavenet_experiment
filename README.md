# wavenet_experiment

Playing with a quick reimplementation of ibab's wavenet.  You may want to
have a look at the readme in [ibab's repo](https://github.com/ibab/tensorflow-wavenet)

- requirements: [librosa](https://github.com/librosa/librosa) for audio.

- Training:  wn_trainer.py
- Generation: wn_generate.py

Training corpus:
   [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (around 10.4GB, [Alternative host](http://www.udialogue.org/download/cstr-vctk-corpus.html))

with VCTK and the default parameters, one epoch of training is about 34000
steps.  You will need to wait an epoch or two before you can expect to see
anything reasonable in generation.

To run this:

./wn_trainer.py -l {logdir} -o {checkpoint-file}

then after training a while:

./wn_generate.py -i {checkpoint-file-#####} -n {n-samples} -o {out.wav}

The default parameters work for me on an 8Gb NVidia GTX-1070.  At the
default parameter settings, it takes 0.9 seconds per time step with
an audio chunk of 50000 samples (half of ibab's default.)  You will
probably want to run 100000 steps, which will take at least a day unless
you have a better GPU card.

If you have a card with smaller memory, you can run with a smaller
audio chunk size.  I carry context correctly across chunks, so there
should be no impact on modeling accuracy.  It does change the effective
batch-size for derivatives, but that might actually be a good thing.

Many parameters must be changed using a parameter file.  Use "-p params.txt".
I include an example file.  It is executable Python.

# Advantages over the ibab setup

- Simpler code. Generation is the same code as training.
- Batch normalization is an option.
- either one-hot or scalar input features, both in generation and training.
  - scalar input seems to produce better results
- I get better accuracy, at least last time I compared.
- You should be able to use n-with atrous convolutions in generation.
  (I haven't tested it.)
- You can train predicting N samples into the future, not just one.

