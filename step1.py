import cv2
import os

dataset_path = "dataset"
categories = ["normal", "leukemia"]

for category in categories:
    path = os.path.join(dataset_path, category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))

        cv2.imshow("Image", img)
        cv2.waitKey(500)  # show each image for 0.5 sec

cv2.destroyAllWindows()