from ultralytics import YOLO
import cv2
from utils import get_car, read_license_plate, write_csv
from sort.sort import *

results = {}

_tracker = Sort()

vehicles = [2,3,5,7]
#load model
coco_model = YOLO('yolov8.pt')
license_plate_detector = YOLO('model/license_plate_detector.pt')
#load video
cap  = cv2.VideoCapture('sample/samle.mp4')

#read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr +=1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        #detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        # print(detections)
        for detection in detection.boxes.data.tolist():
            # print(detection)
            x1, y1, x2, y2, score, class_id = detection    
            if class_id in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                
        #track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        #detect lisense plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxex.data.to_list():
            x1, y1, x2, y2, score, class_id = license_plate
            #assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                #crop license plate
                license_plate_crop = frame[int(x1):int(x2), int(y1):int(y2), :]
                #process plicense plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # cv2.inshow('original_crop', license_plate_crop)
                # cv2.imshow('Processed_crop', license_plate_crop_thresh)
                #read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2] },
                                                'license_plate': {'bbox':[x1, y1, x2, y2],
                                                                    'text': license_plate_text ,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score,}
                                                }
#write the results
            
write_csv(results, './test.csv')