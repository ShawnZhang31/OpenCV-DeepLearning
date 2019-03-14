import cv2
import numpy as np
import dlib

# 使用Dlib获取的面部区域
def getFaceBox(frame):
    frameDlib = frame.copy()
    faceDetector = dlib.get_frontal_face_detector()
    bboxes = []
    faces = faceDetector(frameDlib, 0)
    for face in faces:
        bboxes.append([face.left(), face.top(), face.right(), face.bottom()])
        cv2.rectangle(frameDlib, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2, 8)
    return frameDlib, bboxes

#性别识别模型
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

#年龄识别模型
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']



image = cv2.imread("munv.jpg", cv2.IMREAD_COLOR)
imageFace, bboxes = getFaceBox(image)
padding = 20
if len(bboxes)>0:
    for box in bboxes:
        face = image[max(0,box[1]-padding):min(box[3]+padding,image.shape[0]-1),max(0,box[0]-padding):min(box[2]+padding, image.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # 识别性别
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        cv2.putText(imageFace,str(gender),(min(image.shape[1], box[0]+2), min(image.shape[0], box[3]+20)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1, 8)

        #识别年龄
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        print(agePreds)
        age = ageList[agePreds[0].argmax()]
        cv2.putText(imageFace,str(age),(min(image.shape[1], box[0]+2), min(image.shape[0], box[3]+40)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1, 8)

cv2.imshow("image", imageFace)

cv2.waitKey(0)


