# handball_classifier

## Main purpose

It's should count all statistics for

- passes
- shots
- intercepts

## Plan how to implement it

- [x] Fetch data of few games
- [x] Split into single actions mentioned earlier
- [x] Create dataSet
- [x] Create dataloader
- [x] Create transformation for data
- [x] [WIP] Create CNN
- [x] Create train loop
- [ ] Add some sort of checking how model is working
- [ ] Create CRNN -> CNN LSTM (Long short time memory)
- [ ] Train CRNN
- [ ] Validate on test data

## For future.

- [ ] Use fine-tuning from ImageNet
- [ ] Do some hyperparameter research
- [ ] Validate on full game
- [ ] Check different models
- [ ] Try make loading frames more efficient
- [ ] Maybe use same cache for videos
- [ ] Deploy as app or website

## Resources

- [Medium article](https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031)
- [Brandon Rohrer blog](https://e2eml.school/blog.html#193)
- [Video classifing using 3d ResNet](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
- [Quick and simple action recognition on movie](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
- [Data loader](https://discuss.pytorch.org/t/video-classification-with-cnn-lstm/113413)
- [Formula for shape](https://stats.stackexchange.com/questions/323313/how-to-calculate-output-shape-in-3d-convolution)
- [Again some caluclations for shape](https://cs231n.github.io/convolutional-networks/#conv)
