# Hyper-Bigram
A Custom Bigram model which predicts cars names without any external api based of txt file

I wanted to see if I could build a character-level Bigram model and teach it how to invent brand-new car names. 

### What it actually does
Instead of using a standard dataset, I created a custom `car.txt` file packed with about 50+ exotics, JDMs, and hypercars (some like Senna, Zonda, Skyline, Jesko, etc.). 

The engine:
1. Dynamically reads the text file and builds a character vocabulary.
2. Converts the characters into tensors using one-hot encoding.
3. Runs the forward pass (matrix multiplication & softmax) and calculates the negative log-likelihood loss.
4. Uses backpropagation to manually update a 2D weight matrix until it figures out the spelling patterns.

### The Results
Because it's a Bigram model, it only has a memory of one character (less than a gold fish) at a time when predicting the next one. That means it sometimes spits out absolute genius, and sometimes complete nonsense. 

Here are a few cars the AI actually generated after training:
* **`gtomipraca`** - Honestly sounds like an unreleased Italian hypercar.
* **`mi husky`** - Basically a rugged, off-road rally trim.
* **`po`** - A very known Panda 

### How to play with it
Just clone the repo, make sure you have PyTorch installed, and run the Python file. Make sure the `car.txt` file stays in the exact same folder, or the network won't have any data to train on. 

---
**Built by Priyum.**
*Catch the rest of my coding journey and AI builds over at **experimental16056**.*
