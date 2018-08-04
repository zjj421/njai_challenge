## Notes

### Try

1. 3 channels input. tried, no use.
2. One possible solution would be to train on crops and predict on full images.
 And that segmentation works better when the object is smaller than the input image. 
3. Test time augmentation. try averaging different predictions predicted by different model before postprocessing.

4. train from scrach and train with pretrained encoders.

5. find best low threshold and high threshold between which the predicted mask is unreliable. Sorted all results unreliable picels area, and additionally processed the worst case. For these cases, we selected best-performing models and adjusted probability. 

6. strong scaling augmentation. a lot of zooming in and out and aspect ratio changes before taking the crops.

7. try binary dilation post-processing.

8. Adjusting threshold according to validation set.

### Advices

> [reference here](http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/)
1. Downscale input image, this could lead to some losses in accuracy. 
Since the scores were so close to each oter, I did not want to lose a single pixel on this transformations.

2. When calculating BCE loss, each pixel of the mask was weighted according to 
the distance from the boundary of the object. Pixels on the boundary had 3 times
larger weight than deep inside the area of the object. 
