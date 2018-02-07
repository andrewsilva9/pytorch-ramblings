# pytorch-ramblings

Various files for me to test and learn PyTorch. 

### Basics:

Mostly just files from the regular PyTorch tutorials, in addition to some small stuff. Nothing really of consequence, though useful as reference if I forget something basic.

Todo (Basics):
- [ ] Transform pys into ipynbs for easier interpretation / practice / explanation

### Interruptibility:

Things I explored and played with as potential models for interruptibility. Ultimately, not any more successful than any other algorithm, so the effort was abandoned. Worth re-opening someday if we use end-to-end training on images.

### Style:

Neural style transfer, mostly just copied and adapted tutorial. Played with learning Bob Ross style.

### LSTMs

Various files for learning and practicing LSTMs. Right now, it is applied only to the IMDB and Yelp datasets. Currently, FC-LSTM is in-progress as I explore activation functions, and try to figure out how to properly handle gradients (not working properly according to temporal structure at the moment). Otherwise, lstm_utils handles the training, model declaration, and data loading / processing / cleaning. lstm_test is for imdb, yelp_lstm is for yelp. It was just cleaner / easier to work on that way. Getting mid-80s on IMDB, mid-60s on Yelp.

Todo (LSTM):
- [ ] Better commenting and explanation through ipynbs
- [ ] Sync ipynbs and pys better, currently pys are more developed so that I can run them in a screen remotely
- [ ] Add seq2seq for chatbot
- [ ] Refine FCLSTM
- [ ] Improve README and add dataset links
