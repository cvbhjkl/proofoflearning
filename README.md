## Proof of Directed Guiding Gradients: A New Proof of Learning Consensus Mechanism with Constant-time Verification

Our code relies on the following package:

```
numpy=='1.20.2'
torch=='1.8.1'
torchvision=='0.9.1'
python=='3.8.12'
```

model.py contains two deep learning mode, Resnet18 and MobilenetV2.

pol.py contains some basic function to support Proof of Learning.

To test mining and generate valid proofs of PoL, simply run:

```python
python mining.py
```

To test verification of blocks, use function 

```
block_verify(number) 
```

in mining.py and remove the mining part of the main function to verify the validity of a block.

Note that setting the difficulty too small may result in the failure of verification in checking accuracy.

Our experiment is implemented on two Datasets: CIFAR-10 and MNIST, by default, CIFAR-10 is active. MNIST can be activated by modifying comments.

By default Resnet18 is network structure, MobilenetV2 can be used by modifying related comments.

Some code for counting, such as the code that counts the block interval of all blocks, is placed in the comments section after mining.py


