# handball_classifier

## Tensorflow
Project should be able to classify action's from video. Like:
- passes
- shots
- saves

### Implementation
It was implemented with CNN - LSTM architecture. First all videos are passed to CNN which is pretrained InceptionV3
And after getting features it's passed to recurrent unit - LSTM.
I have gathered data and labeled it by myself.

### Conclusion of project
I have very little data and probably biggest influence in performance has bad distribution of data -> a lot of passes but not so much of saves.

### Second attemp to project 
Because I had too few materials and manually trimming videos was time-consuming I have created video player for classifying videos
-> [Tool](https://github.com/MCygert/video_player_for_classifing). So I will try to gather more videos and tackle the problem once again.




### Things which I would like to try with this project
1. Use different model like Video transformers -> [paper](https://arxiv.org/abs/2103.15691)
2. Create better dataset loader -> [Data api](https://www.tensorflow.org/guide/data_performance)
3. Try saving CNN features to files so I won't have to run them every time.
4. Create confusion matrix of results.
5. Structure project into separate files so it will be more clean (Not only in notebooks)
---





## Pytorch -> I stopped learning Pytorch and changed to tensorflow but I will leave here the progress I made
### Main purpose

It's should count all statistics for

- passes
- shots
- intercepts

### Plan how to implement it

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

### For future.

- [ ] Use fine-tuning from ImageNet
- [ ] Do some hyperparameter research
- [ ] Validate on full game
- [ ] Check different models
- [ ] Try make loading frames more efficient
- [ ] Maybe use same cache for videos
- [ ] Deploy as app or website

### Resources

- [Medium article](https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031)
- [Brandon Rohrer blog](https://e2eml.school/blog.html#193)
- [Video classifing using 3d ResNet](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
- [Quick and simple action recognition on movie](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
- [Data loader](https://discuss.pytorch.org/t/video-classification-with-cnn-lstm/113413)
- [Formula for shape](https://stats.stackexchange.com/questions/323313/how-to-calculate-output-shape-in-3d-convolution)
- [Again some caluclations for shape](https://cs231n.github.io/convolutional-networks/#conv)
