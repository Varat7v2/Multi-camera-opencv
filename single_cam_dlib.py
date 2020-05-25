#!/home/varat/cv/bin/python3

import cv2 
import numpy as np
import tensorflow as tf
import pandas as pd
from imutils import face_utils
import dlib
import time

from faceDetection_frozenGraph import TensoflowFaceDector

PATH_TO_CKPT = 'model/frozen_inference_graph_face.pb'
DLIB_LANDMARKS_MODEL = 'model/shape_predictor_68_face_landmarks.dat'

def main():
    # define a video capture object 
    lcam = cv2.VideoCapture(1)
    # rcam = cv2.VideoCapture(2)
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)
    dlib_landmarks = dlib.shape_predictor(DLIB_LANDMARKS_MODEL)

    lwidth = int(lcam.get(3))
    # lheight = int(lcam.get(4))

    # rwidth = int(rcam.get(3))
    # rheight = int(rcam.get(4))

    # output_vid = cv2.VideoWriter('two_camera_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (lwidth*2,lheight))

    # count = 0
    # pts_src = np.array([
    #                     [553,272],
    #                     [535, 246],
    #                     [204, 267],
    #                     [191, 235],
    #                     [362, 118],
    #                     [112, 239],
    #                     [594, 398]
    #                     ])
    # pts_dst = np.array([
    #                     [1119-640,199],
    #                     [1103-640, 174],
    #                     [770-640, 187],
    #                     [757-640, 157],
    #                     [933-640, 42],
    #                     [676-640, 161],
    #                     [1153-640, 324]
    #                     ])

    # print('Calculating homographic matrix')
    # # calculate the homography
    # H, status = cv2.findHomography(pts_src, pts_dst)

    # box_area = list()
    # box_dist_x = list()
    # box_dist_y = list()
    # rleft_correction = list()
    # rright_correction = list()
    # landmark_x_offset = list()

    while(True): 
        t = time.time()

        ret, lframe = lcam.read()
        # ret, rframe = rcam.read()

        lgray = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
        # rgray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)

        lheight, lwidth, lchannel = lframe.shape
        # rheight, rwidth, rchannel = rframe.shape

        # im_src = lframe
        # im_dst = rframe 

        lboxes, lscores, lclasses, lnum_detections = tDetector.run(lframe)
        lboxes = np.squeeze(lboxes)
        lscores = np.squeeze(lscores)

        # rboxes, rscores, rclasses, rnum_detections = tDetector.run(rframe)
        # rboxes = np.squeeze(rboxes)
        # rscores = np.squeeze(rscores)

        for score, box in zip(lscores, lboxes):
            if score > 0.5:
                # H, status = cv2.findHomography(pts_src, pts_dst)
                lleft = int(box[1]*lwidth)
                ltop = int(box[0]*lheight)
                lright = int(box[3]*lwidth)
                lbottom = int(box[2]*lheight)

                # DLIB FACIAL LANDMARKS
                ldlibRect = dlib.rectangle(lleft, ltop, lright, lbottom)
                t1 = time.time()
                lshape = dlib_landmarks(lgray, ldlibRect)
                print(time.time() - t1, 'secs')

                cv2.rectangle(lframe, (lleft, ltop), (lright, lbottom),(0, 255, 0), int(round(lheight/150)), 8)

                # pts_src = np.array([[lleft, ltop],[lright, ltop],[lleft,lbottom],[lright, lbottom]])
                # pts_src = np.array([[lleft, ltop]])

                # left_top = H.dot(np.array([[lleft], [ltop], [1]]))
                # right_bottom = H.dot(np.array([[lright], [lbottom], [1]]))

                # x1, y1 = int(left_top[0][0]), int(left_top[1][0])
                # x2, y2 = int(right_bottom[0][0]), int(right_bottom[1][0])

                # box_width = x2 - x1
                # box_height = y2 - y1
                # area = box_width * box_height

                # linear_offset_x = int(0.0012629*area+9.8289)
                # linear_offset_x = 1.4E-03*area+6
                # quadratic_offset_x = int(2.77863+0.0022863*area-1.79396E-8*np.power(area,2))

                # cv2.rectangle(lframe, (x1, y1), (x2, y2), (0, 0, 255), int(round(lheight/150)), 8)
                # cv2.rectangle(rframe, (int(x1-linear_offset_x), y1+20), 
                #     (int(x2-linear_offset_x), y2+20), (0, 0, 255), int(round(lheight/150)), 8)
                # cv2.rectangle(rframe, (x1-quadratic_offset_x, y1), (x2-quadratic_offset_x, y2), (255, 0, 0), int(round(lheight/150)), 8)

                # for rscore, rbox in zip(rscores, rboxes):
                #     if rscore > 0.5:
                #         rleft = int(rbox[1]*rwidth)
                #         rtop = int(rbox[0]*rheight)
                #         rright = int(rbox[3]*rwidth)
                #         rbottom = int(rbox[2]*rheight)

                #         corrected_x1 = x1-linear_offset_x
                #         corrected_x2 = x2-linear_offset_x
                #         rleft_correction.append(corrected_x1 - rleft)
                #         rright_correction.append(corrected_x2-rright)

                #         # print(corrected_x1 - rleft, corrected_x2-rright)

                #         offset_x = x1 - rleft
                #         offset_y = y1 - rtop

                #         box_area.append(area)
                #         box_dist_x.append(offset_x)
                #         box_dist_y.append(offset_y)

                #         rdlibRect = dlib.rectangle(rleft, rtop, rright, rbottom)
                #         rshape = dlib_landmarks(rgray, rdlibRect)

                #         cv2.rectangle(rframe, (rleft, rtop), (rright, rbottom), (255,255,255), int(round(rheight/150)), 8)
                        # cv2.rectangle(rframe, (x1, y1), (x2, y2), (0, 0, 255), int(round(lheight/150)), 8)

                        # pts_dst = np.array([[rleft, rtop],[rright, rtop],[rleft,rbottom],[rright, rbottom]])
                        # pts_dst = np.array([[rleft, rtop]])

                for i in range(67):
                    cv2.circle(lframe, (lshape.part(i).x, lshape.part(i).y), 1, (0,0,255), -1)
                    # cv2.circle(rframe, (rshape.part(i).x, rshape.part(i).y), 1, (255,255,255), -1)
                    # transformed_point = H.dot(np.array([[lshape.part(i).x], [lshape.part(i).y], [1]]))
                    # landmark_x_offset.append
                    # cv2.circle(rframe, (int(transformed_point[0][0]-linear_offset_x), int(transformed_point[1][0]+20)), 1, (0,0,255), -1)

        fps = 1/(time.time() - t)
        print(fps)
        # window = np.hstack([lframe, rframe])
        cv2.imshow('Single camera setup', lframe)
        # output_vid.write(window)

        # cv2.imshow('Left Camera', cv2.resize(lframe,(2*lwidth, 2*lheight)))
        # cv2.imshow('Right Camera', cv2.resize(rframe,(2*lwidth, 2*lheight)))

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            # df = pd.DataFrame(list(zip(box_area, box_dist_x, box_dist_y, rleft_correction, rright_correction)), 
            #     columns =['box_area', 'box_dist_x', 'box_dist_y', 'rleft_correction', 'rright_correction'])
            # df.to_csv('two_camera_data.csv') 
            break

    cv2.destroyAllWindows()
    # output_vid.release()

if __name__ == '__main__':
    main()