import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_features(frame):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)

    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    return matches

def crop_image(event, x, y, flags, param):
    coordinates, cropping, first_frame = param

    if event==cv2.EVENT_LBUTTONDOWN:
        coordinates[0] = (x, y)
        cropping[0] = True
        print("Start cropping...")

    elif event == cv2.EVENT_MOUSEMOVE and cropping[0]:
        coordinates[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        coordinates[1] = (x, y)
        cropping[0] = False
        print(f"Image cropped! Template: ({x}, {y})")

def template_crop(frame):
    coordinates = [(0, 0), (0, 0)]
    cropping = [False]
    template = None

    cv2.imshow("crop here", frame)
    cv2.setMouseCallback("crop here", crop_image, [coordinates, cropping, first_frame])

    while True:
        temp_image = frame.copy()

        if cropping[0]:
            cv2.rectangle(temp_image, coordinates[0], coordinates[1], (0, 255, 0), 2)

        cv2.imshow("first image", temp_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # imshow template

    cv2.destroyWindow("first image")
    cv2.destroyWindow("crop here")

    if coordinates[0] != coordinates[1]:
        template = frame[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]

    return template

lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

file_path = 'dataset\\test_data\\dataset(2).mp4'
folder_path = 'dataset\\test_data'

cap = cv2.VideoCapture(file_path)
ret, first_frame = cap.read()
frame_h, frame_w, _ = first_frame.shape

fps = cap.get(cv2.CAP_PROP_FPS)

# define for video saved output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = file_path.split("\\")
save_path = os.path.join(folder_path, f"2_normal_{filename[2]}")
out = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))

prev_frame = first_frame

template = template_crop(first_frame)

cv2.imshow('template', template)
cv2.waitKey(0)

# extract feature
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
kp1, des1 = extract_features(template_gray)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp2, des2 = extract_features(prev_frame_gray)

# match template dan frame
matches = match_features(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)
matches_kp = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

p0 = matches_kp[:10].reshape(-1, 1, 2)

# template - first frame matches
template_matches = cv2.drawMatches(template, kp1, first_frame, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# initialize mask
mask = np.zeros_like(prev_frame)
mask_size = 360

min_kp =  5


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("no frame grabbed")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or len(p1) < min_kp:
        print("cropping a new template")
        template = template_crop(frame)

        cv2.imshow('new template', template)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        kp1, des1 = extract_features(template_gray)

        kp2, des2 = extract_features(frame_gray)

        matches = match_features(des1, des2)
        matches = sorted(matches, key=lambda x:x.distance)
        matches_kp = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        p0 =  matches_kp[:10].reshape(-1, 1, 2)

        mask_size = 400

        continue
        
    else:
        good_new = p1[st==1]
        good_old = p0[st==1]

        x_coords = [pt[0] for pt in good_new]
        avg_x = int(np.mean(x_coords))

        x_start = max(int(avg_x - mask_size//2), 0)
        x_end = min(int(avg_x + mask_size//2), frame_w)
        y_start = 0
        y_end = frame_h

        matched_mask = np.zeros_like(frame)
        matched_mask[y_start:y_end, x_start:x_end] = 255

        masked_frame = cv2.bitwise_and(frame, matched_mask)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(masked_frame, mask)

        out.write(masked_frame)

    prev_frame_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)


    cv2.imshow('out video', img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()