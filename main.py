import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
video_capture = cv2.VideoCapture(0)
ishan_image = face_recognition.load_image_file("faces/ishan.jpg")
ishan_encoding = face_recognition.face_encodings(ishan_image)[0]

elon_image = face_recognition. load_image_file("faces/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

bill_image = face_recognition.load_image_file("faces/bill.jpg")
bill_encoding = face_recognition.face_encodings(bill_image)
known_face_encodings = [ishan_encoding,elon_encoding]
known_face_name = ["ishan","elon","bill"]

students = known_face_name.copy()
face_location = []
face_encoding = []

now = datetime.now()
current_date = now.strftime("%y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_name[best_match_index]

            if name in known_face_name:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomleftcorneroftext = (10,100)
                fontScale =1.5
                fontColor =(255, 0 ,0)
                thickness=3
                linetype =2
                cv2.putText(frame, name +"Present",bottomleftcorneroftext,font,fontScale,fontColor,thickness,linetype)

                if name in students :
                    students.remove(name)
                    current_time =now.strftime("%H-%M%S")
                    lnwriter.writerow([name,current_time])

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1)&0xFF == ord("q"):
                break
                video_capture.release()
                cv2.destroyAllwindows()
