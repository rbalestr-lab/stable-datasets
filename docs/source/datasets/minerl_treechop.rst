MineRL Treechop
==============

``MineRLTreechop`` builds a transition-level dataset from locally acquired
``MineRLTreechop-v0`` demonstrations. It does not download or redistribute
MineRL data. Each row stores an RGB POV image, a deterministically flattened
action vector, reward, terminal flag, episode ID, trajectory name, and timestep.

.. code-block:: python

   from stable_datasets.video import MineRLTreechop

   dataset = MineRLTreechop(
       split="train",
       data_dir="/path/to/minerl-data",
   ).with_format("torch")

   sample = dataset[0]
   # image: C x H x W; action: flattened MineRL controls

The episode and timestep fields preserve trajectory boundaries, allowing
downstream world-model code to construct temporal windows explicitly. Install
MineRL and acquire the source demonstrations under its own terms before use.
