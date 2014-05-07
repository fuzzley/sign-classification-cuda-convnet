Sign Classification With cuda-convnet
================================

This was the final project for CSCE 768, Pattern Recognition.

Dataset:
1. German street signs (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

Prerequisites:
1. Linux OS
2. CUDA 6.0
3. Python, python-numpy, python-magic, python-matplotlib
4. atlaslib
5. cuda-convnet (https://code.google.com/p/cuda-convnet/)

To compile cuda-convnet:
  I included the files necessary to compile cuda-convnet under the "to_compile_cudaconvnet" folder
1. Overwrite build.sh
2. Overwrite Makefile
3. Add common-gcc-cuda-6.0.mk (optionally delete the old "common-gcc-cuda-4.0.mk")
4. Add the "dummyinclude" folder

To create batches:
1. Look over "gen-train-batches.cfg" inside the src folder and make sure the paths for image classes are correct
2. In the terminal, start the python interpreter (just type in "python" and hit enter")
3. Type in the following:
3a. execfile("utils.py")
3b. gen_batches_from_config("gen-train-batches.cfg")
3c. After it finishes, type ctrl + d to exit python interpreter
4. Your batches should now be in the "save_path" specified in "gen-train-batches.cfg"

To train:
1. Modify train.sh with paths to wherever cuda-convnet is installed, where your batches are saved, and which batches are training/testing
2. In the terminal, type, "sh train.sh"
3. To stop the training process at any time, type in ctrl + cfg

To test:
1. Modify test.sh with paths to wherever cuda-convnet is, as well as where the checkpoints are saved
2. In the terminal, type "sh.test.sh"
