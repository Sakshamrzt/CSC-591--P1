"""
Pruning Quickstart
==================
Here is a three-minute video to get you started with model pruning.
..  youtube:: wKh51Jnr0a8
    :align: center
Model pruning is a technique to reduce the model size and computation by reducing model weight size or intermediate state size.
There are three common practices for pruning a DNN model:
#. Pre-training a model -> Pruning the model -> Fine-tuning the pruned model
#. Pruning a model during training (i.e., pruning aware training) -> Fine-tuning the pruned model
#. Pruning a model -> Training the pruned model from scratch
NNI supports all of the above pruning practices by working on the key pruning stage.
Following this tutorial for a quick look at how to use NNI to prune a model in a common practice.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-trained on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Pruning Model`_.

import torch
import time
from torch import nn, optim
from nni.compression.pytorch.speedup import ModelSpeedup
from vae1 import VAE, loss_function, trainer, evaluator, device
from nni.compression.pytorch.pruning import  L1NormPruner, FPGMPruner

# # define the model
config_list1 = [{'sparsity': 0.5, 'op_types': ['Linear']}, {'exclude': True, 'op_names': ['fc4']}]
config_list2 = [{'sparsity': 0.2, 'op_types': ['Linear']}, {'exclude': True, 'op_names': ['fc4']}]
config_list3 = [{'sparsity': 0.8, 'op_types': ['Linear']}, {'exclude': True, 'op_names': ['fc4']}]

config_list = [config_list1, config_list2, config_list3]
pruner_list = [FPGMPruner, L1NormPruner]

for pruner in pruner_list:
    for config in config_list:

        model = VAE().to(device)

        # show the model structure, note that pruner will wrap the model layer.
        print(model)

        # %%

        # define the optimizer and criterion for pre-training
        time_sum = 0
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 10
        time_sum1 = 0
        # pre-train and evaluate the model on MNIST dataset
        for epoch in range(epochs):
            t1 = time.time()
            trainer(model, epoch, optimizer)
            t2 = time.time()
            evaluator(epoch, model)
            time_sum += t2 - t1
            time_sum1 += time.time() - t2

        print("Average training time of unpruned model is ", time_sum / epochs)
        print("Average evaluation time of unpruned model is ", time_sum1 / epochs)

        torch.save(model.state_dict(), "model1.pt")

        # %%
        # Pruning Model
        # -------------
        #
        # Using L1NormPruner to prune the model and generate the masks.
        # Usually, a pruner requires original model and ``config_list`` as its inputs.
        # Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.
        #
        # The following `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
        # except the layer named `fc3`, because `fc3` is `exclude`.
        # The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.

        pruner_ele = pruner(model, config)

        # # compress the model and generate the masks
        _, masks = pruner_ele.compress()
        pruner_ele.show_pruned_weights()
        # show the masks sparsity
        for name, mask in masks.items():
            print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

        pruner_ele._unwrap_model()

        # # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.

        ModelSpeedup(model, torch.rand(128, 1, 28, 28).to(device), masks).speedup_model()

        # optimizer = SGD(model.parameters(), 1e-2)
        torch.save(model.state_dict(), "model2.pt")
        print(model, "is model")

        time_sum = 0
        time_sum1 = 0

        for epoch in range(1):
            t1 = time.time()
            trainer(model, epoch, optimizer)
            t2 = time.time()
            evaluator(epoch, model)
            time_sum += t2 - t1
            time_sum1 += time.time() - t2

        print("Average training time of pruned model is ", time_sum / epochs)
        print("Average evaluation time of pruned model is ", time_sum1 / epochs)
