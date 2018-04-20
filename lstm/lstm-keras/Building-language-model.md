
## Building the LSTM model
- load the text and convert all of the characters to lowercase to reduce the vocabulary that the network must learn.
- convert char to integer using some mapping (Karpathy)

### Training Small LSTM
- the book text is split into subsequences with a fixed length of 100 characters
- each training pattern of the network is comprised of 100 time steps of one character (X) followed by one character output (y). these sequences are created by sliding this window along the whole book one character at a time, allowing each character a chance to be learned from the 100 characters that preceded it
- transform the list of input sequences into the form [samples, time steps, features] expected by an LSTM network
- rescale the integers to the range 0-to-1 to make the patterns easier to learn by the LSTM network that uses the sigmoid activation function by default.
- convert the output patterns (single characters converted to integers) into a one hot encoding. this allows the network to predict the probability of each of the characters in the vocab instead of predicting exactly the next char.


Input layer is a single hidden LSTM layer with 256 memory units. The network uses dropout with a probability of 0.2. 
The output layer is a Dense layer using the softmax activation function to output a probability prediction for each of the characters between 0 and 1.
Using mini batches of 128 patterns
Training time: ~5 min per epoch
loss = 2.9733 to 2.0477

### Training Large LSTM
- Add a second layer of 256 units with a dropout of 0.2
- decrease batch size to 64 (!explore!)
- increase num of epochs to 50
- training time: ~21 mins per epoch
- loss: 2.8166 to 1.3837


- char level
    - spell checker for words
    - freq of short words in human authored text vs generated
    - bi gram and n-gram

- hyperparameters of the lstm in the keras 
- extrapolate (change size of dataset)
- measuring novelty of the text (edit-distance)







(IM): Split into sentences, pad shorter sentences and truncate longer ones, still using chars
for each char except the first 100, look at the previous 100 chars
Predict fewer than 1,000 characters as output for a given seed.
Remove all punctuation from the source text, and therefore from the models’ vocabulary.
Try a one hot encoded for the input sequences.
Train the model on padded sentences rather than random sequences of characters.
Increase the number of training epochs to 100 or more.
Add dropout to the visible input layer and consider tuning the dropout percentage.
Tune the batch size, try a batch size of 1 as a baseline and larger sizes.
Add more memory units to the layers and/or more layers.

Acronyms

The Unreasonable Effectiveness of Recurrent Neural Networks, Karpathy.
Generating Text with Recurrent Neural Networks, Sutskever, 2011
Keras code example of LSTM for text generation <https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py>
Lasagne code example of LSTM for text generation <https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py>