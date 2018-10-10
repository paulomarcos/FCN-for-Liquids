import cv2
import os
import numpy as np

image_folder  = 'runs/batch_1_epoch_10'
image_folder2 = 'runs/batch_10_epoch_20'
video_name = 'video.avi'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc, 24, (width*2,height))

font = cv2.FONT_HERSHEY_SIMPLEX

for image in sorted(images):

    frame1 = cv2.imread(os.path.join(image_folder, image))
    cv2.putText(frame1, image_folder,(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)
    frame2 = cv2.imread(os.path.join(image_folder2, image))
    cv2.putText(frame2,image_folder2,(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    result = np.concatenate((frame1, frame2), axis=1)

    video.write(result)

cv2.destroyAllWindows()
video.release()
print("Video saved as " + video_name)
