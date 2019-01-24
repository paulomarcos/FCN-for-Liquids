import cv2
import os
import numpy as np

image_folder  = 'runs/keio_nirct_nonir_25epochs'
image_folder2 = 'runs/keio_nirct_withnir_25epochs'
video_name = 'comparison_video_slow.avi'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

accuracy = (("data_3",0.9344,0.9524),
            ("data_2",0.9648,0.9764),
            ("data_1",0.9105,0.9488),
            ("data_6",0.4301,0.5016),
            ("data_3_flipped",0.924,0.9358),
            ("data_4",0.9447,0.9499),
            ("data_1_flipped",0.9385,0.9465),
            ("data_4_flipped",0.9499,0.9736),
            ("data_7",0.744,0.2417),
            ("data_2_flipped",0.9805,0.9849))


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
SIZE = (400, 400)
height = SIZE[0]
width = SIZE[1]
video = cv2.VideoWriter(video_name, fourcc, 4, (width*3,height*2))

font = cv2.FONT_HERSHEY_SIMPLEX
idx = 0

gt_path = "data/data_liquid/testing/"

print("Video encoding...")

for image in sorted(images):

    # Do not want to show ground truth comparison
    if ("result" in image):
        continue

    # Index incrementation for accuracy
    if (idx != int(image[:1])):
        idx += 1

    # Adjust first frame
    frame1 = cv2.imread(os.path.join(image_folder, image))
    frame1 = cv2.resize(frame1,SIZE)
    txt = "Acc for "+accuracy[idx][0]+": "+str(accuracy[idx][1])
    cv2.putText(frame1, txt, (10,60), font, 0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame1, image_folder,(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Adjust second frame
    frame2 = cv2.imread(os.path.join(image_folder2, image))
    frame2 = cv2.resize(frame2,SIZE)
    txt = "Acc for "+accuracy[idx][0]+": "+str(accuracy[idx][2])
    cv2.putText(frame2,image_folder2,(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame2,txt,(10,60), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Adjust contour frame
    contour_img_path = accuracy[idx][0]+"/contour_"+image[-8:]
    frame3 = cv2.imread(os.path.join(gt_path, contour_img_path))
    frame3 = cv2.resize(frame3,SIZE)
    cv2.putText(frame3,"Ground truth contour",(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Adjust fourth frame: predition of frame without nir
    frame4 = cv2.imread(os.path.join(image_folder, "result_"+image))
    frame4 = cv2.resize(frame4,SIZE)
    cv2.putText(frame4,"Prediction without NIR",(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Adjust fourth frame: predition of frame witht nir
    frame5 = cv2.imread(os.path.join(image_folder2, "result_"+image))
    frame5 = cv2.resize(frame5,SIZE)
    cv2.putText(frame5,"Prediction with NIR",(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Adjust fourth frame: predition of frame without nir
    gt_img_path = accuracy[idx][0]+"/ground_truth_"+image[-8:]
    frame6 = cv2.imread(os.path.join(gt_path, gt_img_path))
    frame6 = cv2.resize(frame6,SIZE)
    cv2.putText(frame6,"Ground truth ",(10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

    # Concatenating the two frames
    result_1 = np.concatenate((frame1, frame2), axis=1)
    result_1 = np.concatenate((result_1, frame3), axis=1)

    result_2 = np.concatenate((frame4, frame5), axis=1)
    result_2 = np.concatenate((result_2, frame6), axis=1)

    result = np.concatenate((result_1, result_2), axis=0)
    #print(result.shape, result_1.shape, result_2.shape)

    cv2.imshow("Result screen", result)
    cv2.waitKey(1)

    if (np.sum(result) != 0):
        video.write(result)

video.release()
cv2.destroyAllWindows()
print("Done.")
print("Video saved as " + video_name)
