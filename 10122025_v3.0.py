import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def path_quadratic(m, g, b2, t):
    vT = np.sqrt(m * g / b2)
    return (vT**2 / g) * np.log(np.cosh(g * (t-t[0]) / vT))


def video(m, g, min_bound, max_bound, fname, distance,thrmin,thrmax,kersize):

    cap = cv2.VideoCapture(fname)
    namestr = os.path.splitext(os.path.basename(fname))[0]

    frames = []
    centers_y_meters = []
    times = []

    frame_id = 0
    fps = 60

    while True:
        ret, frame = cap.read()
        if not ret:
            break
#Image processing:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thrmin, thrmax, cv2.THRESH_BINARY_INV)

        kernel = np.ones((kersize, kersize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            frame_id += 1
            continue

        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            frame_id += 1
            continue

        cy = int(M["m01"] / M["m00"])

        image_height_px = frame.shape[0]
        meters_per_pixel = distance / image_height_px
        y_meters = cy * meters_per_pixel

        frames.append(frame_id)
        centers_y_meters.append(y_meters)
        times.append(frame_id / fps)

        frame_id += 1
#---------------------
#numerical part
    cap.release()
    t = np.array(times[min_bound:max_bound])
    y = np.array(centers_y_meters[min_bound:max_bound])

    #y = savgol_filter(y, 11, 3) unused, bad

    #Numerical Euler mahinācijas
    dt = t[1] - t[0]

    v = (y[1:] - y[:-1]) / dt
    a = (v[1:] - v[:-1]) / dt

    t_mid = t[2:]

    # -----------------------------
    #b2 from data
    # -----------------------------
    b2_list = []

    for k in range(1, 10):   #take only the last frames (might have to change the range upper value in case of low frame count)
        v_tail = v[-k:]
        a_tail = a[-k:]

        mask = v_tail > 0.1
        if np.sum(mask) < 3:
            continue

        b2_values = m * (g - a_tail[mask]) / (v_tail[mask]**2)
        b2_est = np.mean(b2_values)

        b2_list.append(b2_est)

    b2_mean = np.mean(b2_list)

    print(f"Estimated quadratic drag coefficient b2 = {b2_mean:.4e} kg/m")

    plt.figure()
    plt.plot(t, y, "o", label="Experiment")
    
    plt.plot(t,
             path_quadratic(m, g, b2_mean, t),
             label="Quadratic drag model")

    plt.plot(t,
             0.5 * g * (t-t[0])**2,
             "--",
             label="Free fall")

    plt.xlabel("Time [s]")
    plt.ylabel("Vertical position [m]")
    plt.title("Trajectory comparison for {}".format(namestr[:-3])) #might have to tamper with .format(namestr...) if the title is not looking good
    plt.legend()
    plt.grid(True)

    plt.ylim(distance, 0)
    plt.savefig("{}_quadratic_drag.png".format(namestr[:-2]), dpi=300)
    plt.show()

    return b2_mean


#object--mass[kg]
#normal ball: 0.045
#the smaller disk: 0.014
#medium disk: 0.027
#the bigger disk: 0.061
#hemisphere: 0.021
#maisiņš: 0.018
#foam plastic ball 0.00183



#video(0.021,9.81,0,-5,"C:/Users/raulss-/Documents/pretkritiens4.mp4",2.3)
#video(0.027,9.81,24,-16,"vripa2(2).mp4",2.4,130,255,15)


#video(0.061,9.81,0,-17,"lripa6.mp4",2.5,130,255,15)


#video(0.027,9.81,30,-22,"mripa3(1).mp4",2.5,100,255,5)

video(0.021,9.81,10,-5,"pretkritiens6(1).mp4",2.5,120,255,7)


