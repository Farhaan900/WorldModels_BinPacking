# WorldModelsDreaming

(Code and Readme, from: https://github.com/hardmaru/WorldModelsExperiments/, https://github.com/awslabs/or-rl-benchmarks)

Step by step instructions of reproducing BinPacking experiment.

# Notes

-TensorFlow 1.8.0

-NumPy 1.13.3 (1.14 has some annoying warning)

-OpenAI Gym 0.9.4 (have not tested 1.0+, big gym api changes)

-cma 2.2.0, basically 2+ should work

-Python 3, although 2 might work.

-mpi4py 2 (see https://github.com/hardmaru/estool)

# Reading

https://worldmodels.github.io/

http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

# Instructions for running the pretrained model already in repo

Play using your keyboard (note: set mac os x screen resolution to highest). Make sure you have set the variables self.record=False and self.should_render=True:

`python bin_packing_environment.py`

# Instructions for training everything from scratch

Extract 10k random episodes by running the following:

`python run_evaluation_dopamine.py --base_dir="test_results" --gin_files="test_configs/rainbow.gin"`

For this make sure you have set the environment (bin_packing_environment) self.record=True and self.should_render=False.

After running this, 12.8k episodes should be saved as npz files in `record`. We will only use 10k episodes.

On a machine with a GPU, run the following to train VAE for 10 epochs:

`python vae_train.py`

After training, the vae model saved in `tf_vae/vae.json`

Next, we pre-process collected data using pre-trained VAE by running:

`python series.py`

A new dataset will be created in `series`. After this is recorded, train the MDN-RNN by running:

`train rnn using python rnn_train.py`

This will produce a model in `tf_rnn/rnn.json` and also `initial_z.json`.

You must now copy copy vae.json, initial_z.json and rnn.json over to vae, initial_z, and rnn directories respectively, and overwrite previous files if they were there.

Now on a 64-core CPU machine, run the CMA-ES based training:

`python train.py`

You can monitor progress using the `plot_training_progress.ipynb` notebook which loads the `log` files being generated. You can test the model by running:

`python model.py norender log/carracing.cma.16.64.best.json`

# Citation

If you find this project useful in an academic setting, please cite:

```
@article{Ha2018WorldModels,
  author = {Ha, D. and Schmidhuber, J.},
  title  = {World Models},
  eprint = {arXiv:1803.10122},
  doi    = {10.5281/zenodo.1207631},
  url    = {https://worldmodels.github.io},
  year   = {2018}
}
```

# Licence

MIT
