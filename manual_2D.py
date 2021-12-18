import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, isdir, join
import os.path
import csv

class relative_calibration():
    def __init__(self):  # Run it once
        # Path
        self.video_path = "./input/crosswalk_front.jpg"  # calibrate video path
        # self.video_path = "../../20190928/cube/L_cube2_undistortion.mp4"
        self.frame = 2050
        # self.video_path = "../../demo/20190815/relative_calibration/L_undistortion.mp4"  # calibrate video path
        self.ratio_h = 1
        self.ratio_w = 1

    def check_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 1, (255, 0, 0), -1)
            self.draw_point.append([[x, y]])
            if len(self.draw_point) > 1:
                cv2.line(self.frame, tuple(self.draw_point[len(self.draw_point)-1][0]), tuple(self.draw_point[len(self.draw_point)-2][0]), (0, 255, 0), 2)
            x = round(x/self.ratio_w)
            y = round(y/self.ratio_h)
            print('Add net point: ( ', x, ', ', y, ' )')
            self.corners.append([[x, y]])


    def Run(self):

        self.frame = cv2.imread(self.video_path)
        h, w = self.frame.shape[:2]
        print("Load the video:：", self.video_path, '(h , w) = ', '(', h, ' , ', w, ')')
        self.frame = cv2.resize(self.frame, None, fx=self.ratio_w, fy=self.ratio_h, interpolation=cv2.INTER_AREA)

        # Manual
        print("Automatic loading failed, please select manually...")
        cv2.namedWindow('Manual')
        cv2.setMouseCallback('Manual', self.check_circle)
        self.corners = []
        self.draw_point = []
        while (1):
            cv2.imshow('Manual', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        new_tilt_coner = []
        for point in self.corners:
            new_tilt_coner.append(point[0])

        with open(os.path.splitext(self.video_path)[0]+'_block.csv', 'w', newline='') as csvfile:
            csv_f = csv.writer(csvfile)
            CSV1 = ['new_tilt_coner']
            for point in new_tilt_coner:
                CSV1.append(point[0])
                CSV1.append(point[1])
            csv_f.writerow(CSV1)


if __name__ == '__main__':
    relative_calibration().Run()