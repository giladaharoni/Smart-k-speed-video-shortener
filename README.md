# Smart-k-speed-video-shortener
A smart way to speed up videos, by calculating the information gain of every new frame compare to the previous, and then selecting the top k information-gaining frames, based on the k - maximal subset problem.
## how to run it
```
smartspeedup.py input_video.mp4 output.mp4 speedup_factor method draw_playbar
```
when:
- **input_video.mp4, output.mp4** are the input and the output files paths, must be in mp4 format.
- **speedup_factor**: how many times to shorten the video
- **method**: should be 1, 2, or 3. 1 - euclidean distance, 2 - skeletonized and Hausdorff distance, 3 - BRIET model output distance.
- **darw_playbar**: True or False, write to the output video the playbar and show which frames were selected.

## More about the methods:
- **Euclidean distance**: Very sensitive to any change of light or color, would capture any sudden movements.
- **Skeleton preprocessing and Hausdorff distance**: Skeletonized frame is represented by objects outline, and Hausdorf distance evaluates the difference in graphs on the metric space. woould capture transformations and scaling of objects, less color changes.
- **BRIET Model output distance**: The BRIET outputs object detection score, represented by logits, and can identifiy 22k types of objects. would capture apearance or disapearance of new objects.
## Example:


