# wavenet_experiment

Playing with a quick reimplementation of ibab's wavenet.  You may want to
have a look at the readme in [ibab's repo](https://github.com/ibab/tensorflow-wavenet)

requirements: [librosa](https://github.com/librosa/librosa) for audio.

Training:  wn_trainer.py
Generation: wn_generate.py

Training corpus:
   [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) (around 10.4GB, [Alternative host](http://www.udialogue.org/download/cstr-vctk-corpus.html))

with VCTK and the default parameters, one epoch of training is about 17000
steps.  You will need to wait 4 or 5 epochs before generation starts to do
anything.

