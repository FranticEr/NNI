import mediapipe as mp
import cv2
import os
import numpy as np


def getFaceImage(image:str|np.ndarray):
    mp_face_detection = mp.solutions.face_detection
    # 加载图像
    if isinstance(image,str):
        image = cv2.imread(image)
        # print("path")
    # 使用Face Detection模块
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
    # 处理检测结果
        if results.detections:
            for detection in results.detections:
            # 获取人脸位置和置信度
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]), \
                         int(bbox.width * image.shape[1]), int(bbox.height * image.shape[0])
                confidence = detection.score
            # 绘制边界框
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)       
            # 截取人脸
                face_image = image[y-20:y+h, x-10:x+w+10]
                # if face_image==None:
                #     print('None')
                return face_image
            


def getFaceFrames(saveFolder, saveFileName, videoFullName,step=4):
    vc=cv2.VideoCapture(videoFullName)
    fps = vc.get(cv2.CAP_PROP_FPS)
    timeF=step*fps
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    if vc.isOpened():
        rval,frame=vc.read()  
    c=1
    while rval:
        try:
            # print(c)
            rval,frame=vc.read()
            if (c%timeF==0):
                frame=getFaceImage(frame)
                cv2.imwrite(os.path.join(saveFolder,saveFileName+f'_{str(c)}.jpg'),frame)
            c=c+1
        except Exception as e:
            print("Exception:",e)
    vc.release()



# for videoName in videoFilenameList:
#     ID_LEVEL,type=videoName.split(".")
#     ID,LEVEL=ID_LEVEL.split('-')    
#     saveFolder=os.path.join(r"D:\project_meta\NNproject\NNI\output\video_frames\FaceLEVELFolder",f'{LEVEL}',f'{ID}')
#     saveFileName=f'{ID}'
#     videoFullName=os.path.join(videoFolder,videoName)
#     getFaceFrames(saveFolder, saveFileName, videoFullName)

def getFrames(saveFolder, saveFileName, videoFullName,step=4):
    vc=cv2.VideoCapture(videoFullName)
    fps = vc.get(cv2.CAP_PROP_FPS)
    timeF=step*fps
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    if vc.isOpened():
        rval,frame=vc.read()  
    c=1
    while rval:
        try:
            # print(c)
            rval,frame=vc.read()
            if (c%timeF==0):
                cv2.imwrite(os.path.join(saveFolder,saveFileName+f'_{str(c)}.jpg'),frame)
            c=c+1
        except Exception as e:
            print("Exception:",e)
    vc.release()


    