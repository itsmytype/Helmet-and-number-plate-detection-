Import cv2

Import numpy as np

Import os

Import imutils

From tensorflow.keras.models import load_model

Os.environ[‘TF_FORCE_GPU_ALLOW_GROWTH’] = ‘true’

Net = cv2.dnn.readNet(“yolov3-custom_7000.weights”, “yolov3-custom.cfg”)

Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

Model = load_model(‘helmet-nonhelmet_cnn.h5’)

Print(‘model loaded!!!’)

Cap = cv2.VideoCapture(‘video.mp4’)

COLORS = [(0,255,0),(0,0,255)]

Layer_names = net.getLayerNames()

Output_layers = [layer_names[I – 1] for I in net.getUnconnectedOutLayers()]

 

Fourcc = cv2.VideoWriter_fourcc(*”XVID”)

Writer = cv2.VideoWriter(‘output.avi’, fourcc, 5,(888,500))

Def helmet_or_nohelmet(helmet_roi):

 Try:

  Helmet_roi = cv2.resize(helmet_roi, (224, 224))

  Helmet_roi = np.array(helmet_roi,dtype=’float32’)

  Helmet_roi = helmet_roi.reshape(1, 224, 224, 3)

  Helmet_roi = helmet_roi/255.0

  Return int(model.predict(helmet_roi)[0][0])

 Except:

   Pass

Ret = True

While ret:

    Ret, img = cap.read()

    Img = imutils.resize(img,height=500)

    # img = cv2.imread(‘test.png’)

    Height, width = img.shape[:2]

    Blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    Net.setInput(blob)

    Outs = net.forward(output_layers)

    Confidences = []

    Boxes = []

    classIds = []

    for out in outs:

        for detection in out:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.3:

                center_x = int(detection[0] * width)

                center_y = int(detection[1] * height)

                w = int(detection[2] * width)

                h = int(detection[3] * height)

                x = int(center_x – w / 2)

                y = int(center_y – h / 2)

                boxes.append([x, y, w, h])

                confidences.append(float(confidence))

                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for I in range(len(boxes)):

        if I in indexes:

            x,y,w,h = boxes[i]

            color = [int© for c in COLORS[classIds[i]]]

            # green ( bike

            # red ( number plate

            If classIds[i]==0: #bike

                Helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]

            Else: #number plate

                X_h = x-60

                Y_h = y-350

                W_h = w+100

                H_h = h+100

                Cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)

                # h_r = img[max(0,(y-330)):max(0,(y-330 + h+100)) , max(0,(x-80)):max(0,(x-80 + w+130))]

                If y_h>0 and x_h>0:

                    H_r = img[y_h:y_h+h_h , x_h:x_h +w_h]

                    C = helmet_or_nohelmet(h_r)

                    Cv2.putText(img,[‘helmet’,’no-helmet’][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                

                    Cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)

    Writer.write(img)

    Cv2.imshow(“Image”, img)

    If cv2.waitKey(1) == 27:

        Break

Writer.release()

Cap.release()

Cv2.waitKey(0)

Cv2.destroyAllWindows()
