import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
import threading

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector

employee_lock = threading.Lock()
employee_name_recon = None
unknown_employee_name = "UNKNOWN"

def analyze_image(image, input_shape, data_frame, detected_faces_final, enable_face_analysis, face_model, face_model_threshold, emotion_model, age_model, gender_model):
    global employee_name_recon
    global unknown_employee_name

    time.sleep(1)
    for detected_face in detected_faces_final:
        x = detected_face[0]
        y = detected_face[1]
        w = detected_face[2]
        h = detected_face[3]

        # -------------------------------

        # apply deep learning for custom_face

        custom_face = image[y:y + h, x:x + w]

        # -------------------------------
        # facial attribute analysis

        if enable_face_analysis:

            gray_img = functions.preprocess_face(img=custom_face, target_size=(48, 48), grayscale=True,
                                                 enforce_detection=False, detector_backend='opencv')
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_predictions = emotion_model.predict(gray_img)[0, :]
            sum_of_predictions = emotion_predictions.sum()

            mood_items = []
            for i in range(0, len(emotion_labels)):
                mood_item = []
                emotion_label = emotion_labels[i]
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                mood_item.append(emotion_label)
                mood_item.append(emotion_prediction)
                mood_items.append(mood_item)

            emotion_df = pd.DataFrame(mood_items, columns=["emotion", "score"])
            emotion_df = emotion_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

            # background of mood box

            for index, instance in emotion_df.iterrows():
                emotion_label = "%s " % (instance['emotion'])
                emotion_score = instance['score'] / 100

            # -------------------------------

            face_224 = functions.preprocess_face(img=custom_face, target_size=(224, 224),
                                                 grayscale=False, enforce_detection=False,
                                                 detector_backend='opencv')

            age_predictions = age_model.predict(face_224)[0, :]
            apparent_age = Age.findApparentAge(age_predictions)

            # -------------------------------

            gender_prediction = gender_model.predict(face_224)[0, :]

            if np.argmax(gender_prediction) == 0:
                gender = "W"
            elif np.argmax(gender_prediction) == 1:
                gender = "M"

            # print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

            analysis_report = str(int(apparent_age)) + " " + gender

            print(f"employee analysis: emotion: {emotion_label} ({emotion_score}), {analysis_report}")

        # -------------------------------
        # face recognition

        custom_face = functions.preprocess_face(img=custom_face,
                                                target_size=(input_shape[1], input_shape[0]),
                                                enforce_detection=False, detector_backend='opencv')

        # check preprocess_face function handled
        if custom_face.shape[1:3] == input_shape:
            if data_frame.shape[0] > 0:  # if there are images to verify, apply face recognition
                img1_representation = face_model.predict(custom_face)[0, :]

                # print(freezed_frame," - ",img1_representation[0:5])

                def findDistance(row):
                    distance_metric = row['distance_metric']
                    img2_representation = row['embedding']

                    distance = 1000  # initialize very large value
                    if distance_metric == 'cosine':
                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean':
                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                             dst.l2_normalize(img2_representation))

                    return distance

                data_frame['distance'] = data_frame.apply(findDistance, axis=1)
                data_frame = data_frame.sort_values(by=["distance"])

                candidate = data_frame.iloc[0]
                employee_name = candidate['employee']
                best_distance = candidate['distance']

                # print(candidate[['employee', 'distance']].values)

                # if True:
                if best_distance <= face_model_threshold:
                    # print(employee_name)
                    display_img = cv2.imread(employee_name)

                    label = employee_name.split("/")[-2]
                    employee_lock.acquire()
                    employee_name_recon = label
                    employee_lock.release()
                    print(f"employee recognized: {label}")
                    # publish something here
                else:
                    employee_lock.acquire()
                    employee_name_recon = unknown_employee_name
                    employee_lock.release()


def analysis(db_path, model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine',
             enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5):
    # ------------------------

    face_detector = FaceDetector.build_model(detector_backend)
    print("Detector backend is ", detector_backend)

    # ------------------------

    input_shape = (224, 224)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]

    text_color = (255, 255, 255)

    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    # print(exact_path)
                    employees.append(exact_path)

    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path, ") . Face recognition will not be performed.")

    # ------------------------

    if len(employees) > 0:
        model = DeepFace.build_model(model_name)
        print(model_name, " is built")

        # ------------------------

        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0];
        input_shape_y = input_shape[1]

        # tuned thresholds for model and metric pair
        threshold = dst.findThreshold(model_name, distance_metric)

    # ------------------------
    # facial attribute analysis models

    if enable_face_analysis == True:
        tic = time.time()

        emotion_model = DeepFace.build_model('Emotion')
        print("Emotion model loaded")

        age_model = DeepFace.build_model('Age')
        print("Age model loaded")

        gender_model = DeepFace.build_model('Gender')
        print("Gender model loaded")

        toc = time.time()

        print("Facial attibute analysis models loaded in ", toc - tic, " seconds")

    # ------------------------

    # find embeddings for employee list

    tic = time.time()

    # -----------------------

    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

    # TODO: why don't you store those embeddings in a pickle file similar to find function?

    embeddings = []
    # for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
        embedding = []

        # preprocess_face returns single face. this is expected for source images in db.
        img = functions.preprocess_face(img=employee, target_size=(input_shape_y, input_shape_x),
                                        enforce_detection=False, detector_backend=detector_backend)
        img_representation = model.predict(img)[0, :]

        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric

    toc = time.time()

    print("Embeddings found for given data set in ", toc - tic, " seconds")

    # -----------------------

    pivot_img_size = 112  # face recognition result image

    # -----------------------

    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam

    while (True):
        ret, img = cap.read()

        if img is None:
            break

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution = img.shape;
        resolution_x = img.shape[1];
        resolution_y = img.shape[0]

        top_offset = 50
        ellipse_y = int((resolution_y - top_offset) * 0.45)
        ellipse_x = int(ellipse_y * 0.85)
        ellipse_center = (int(resolution_x / 2), int((resolution_y + top_offset) / 2))

        if not freeze:
            # faces = face_cascade.detectMultiScale(img, 1.3, 5)

            # faces stores list of detected_face and region pair
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=False)

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        center_offset = 40
        for face, (x, y, w, h) in faces:
            if ellipse_x * 1.5 < w < ellipse_x * 2 and \
                    ellipse_center[0] - center_offset < x + w / 2 < ellipse_center[0] + center_offset and \
                    ellipse_center[1] - center_offset < y + h / 2 < ellipse_center[
                1] + center_offset:  # discard small detected faces

                face_detected = True
                if face_index == 0:
                    face_included_frames = face_included_frames + 1  # increase frame for a single face

                #cv2.rectangle(img, (x, y), (x + w, y + h), (67, 67, 67), 1)  # draw rectangle to main image

                #cv2.putText(img, str(frame_threshold - face_included_frames), (int(x + w / 4), int(y + h / 1.5)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

            # -------------------------------------

        if face_detected and face_included_frames >= frame_threshold and not freeze:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()

            print("starting thread")
            t = threading.Thread(target=analyze_image, args=(base_img, input_shape, df, detected_faces_final, enable_face_analysis, model, threshold, emotion_model, age_model, gender_model,))
            t.start()

            print("continuing execution")
            tic = time.time()

        if freeze:

            toc = time.time()
            if (toc - tic) < time_threshold:

                time_left = int(time_threshold - (toc - tic) + 1)

                #cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                #cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                display_name = None
                employee_lock.acquire()
                display_name = employee_name_recon
                employee_lock.release()

                if display_name and display_name != unknown_employee_name:
                    message = f"Welcome {display_name}"
                    cv2.putText(img, message, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (50, 50, 50), 2)
                    cv2.ellipse(img=img, center=ellipse_center,
                                axes=(ellipse_x, ellipse_y), angle=0, startAngle=0, endAngle=360,
                                color=(0, 128, 0), thickness=2)
                elif display_name and display_name == unknown_employee_name:
                    cv2.putText(img, "Intruder!!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 9, 200), 2)
                    cv2.ellipse(img=img, center=ellipse_center,
                                axes=(ellipse_x, ellipse_y), angle=0, startAngle=0, endAngle=360,
                                color=(0, 0, 200), thickness=2)
                else:
                    cv2.ellipse(img=img, center=ellipse_center,
                                axes=(ellipse_x, ellipse_y), angle=0, startAngle=0, endAngle=360,
                                color=(0, 128, 0), thickness=2)


                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0

        else:
            cv2.putText(img, "Place your face inside the circle", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 50, 50), 2)
            cv2.ellipse(img=img, center=ellipse_center,
                        axes=(ellipse_x, ellipse_y), angle=0, startAngle=0, endAngle=360,
                        color=(128, 128, 128), thickness=2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()
