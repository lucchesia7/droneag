### Image Processing

The output images from the unsupervised segmentation are post-processed using image processing techniques to mask out all the different regions present in the output image.

### Methods
- Connected Components
Connected component works just like contours. It can be used to find different regions or objects in the image. This can work for the problem we are trying to solve. since we our model has already segments the image, using connected component can work.

### How to Use
- use one of the models weights we already have in our repo to segment an image.
- use this block code to apply connected componet.


```
import cv2
import numpy as np

image = np.array(Image.open("your output image"))
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
image_stats = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

num_labels,labels,stats, _ = image_stats

for i in range(0, num_labels):
    #extract the individual parameters 
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    print(x, y, w, h, area)
    
    mask = (labels == i).astype("uint8") * 255
    
    cv2.imshow("Output", output)
    cv2.imshow("Component", mask)
    
    k= cv2.waitKey(0)
    if k == ord("q"):
        break
cv2.destroyAllWindows()
```