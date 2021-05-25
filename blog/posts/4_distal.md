---
title: Distance
display: home
image: https://render.fineartamerica.com/images/rendered/search/phone-case/iphone12pro/images-medium-5/wireframe-geometrical-models-robert-brookscience-photo-library.jpg
date: 2020-02-20
tags: 
  - display
categories:
  - futurama
--- 

# __Tính khoảng cách giữ vật thể và camera dùng openCV__

## __1. Kiến thức sử dụng:__

#### a. Toán học:
Dùng tính chất đồng dạng trong hình học: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\Delta AHC  \sim  \Delta  CHB\\"/>



![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Pythagoras_similar_triangles_simplified.svg/320px-Pythagoras_similar_triangles_simplified.svg.png)


Dự vào tính chất động dạng ta có biểu thức sau: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\alpha = \frac{AH}{CH} = \frac{HC}{HB} = \frac{AC}{CB} 
\Rightarrow CH = \frac{AC \times HB }{CB}"/>

Trong đó $\alpha$ là một hằng số. 

Đồng dạng vật thể theo tỷ lệ:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/SimilitudeHomoth%C3%A9tieL.svg/320px-SimilitudeHomoth%C3%A9tieL.svg.png)

Dựa vào tính chất:___"Nếu ta biết kích thước của chứ L màu xanh dương và một cạnh của chữ L màu đen ta có thể tính được các cạnh còn lại của chữ L màu đen"___.


Dựa vào câu nhận định trên: 


#### __Ta phải lấy được kích thước cơ bản của vật mẫu và khoảng cách từ vật mẫu đến điểm quan sát làm dữ liệu ban đầu__.

Tham khảo:
1. https://en.wikipedia.org/wiki/Similarity_(geometry)
2. https://vi.wikipedia.org/wiki/%C4%90%E1%BB%93ng_d%E1%BA%A1ng
3. https://en.wikipedia.org/wiki/Similarity_system_of_triangles

b. Kiến thức cơ bản trong sử lý ảnh:

- [Độ phân giải của ảnh](https://en.wikipedia.org/wiki/Image_resolution).
- [Màu sắc của ảnh](https://en.wikipedia.org/wiki/Color_image).
- Một số đơn vị cần chứ ý: 1 pixel = 0.0264583333 cm

## __2. Code thực hiện:__

#### __Nhập thư viện cv2 và PIL để chạy__


```python
import cv2
from PIL import Image
# variables
# distance from camera to object(face) measured
Known_distance = 40.2  # Centimeter
# mine is 14.3 something, measure your face width, are google it
Known_width = 14.3  # Centimeter

# Colors  >>> BGR Format(BLUE, GREEN, RED)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX
```

### __Các hàm sử dụng trong face__


```python
def FocalLength(measured_distance, real_width, width_in_rf_image):
    # Function Description (Doc String)
    '''
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using 
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE 
    :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
 
    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector) 
    :retrun Focal_Length(Float):
    '''
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length
```


```python
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    '''
    This Function simply Estimates the distance between object and camera using arguments(Focal_Length, Actual_object_width, Object_width_in_the_image)
    :param1 Focal_length(float): return by the Focal_Length_Finder function
 
    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated  
 
    '''
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance
```


```python
def face_data(image, CallOut, Distance_level):
    '''
 
    This function Detect face and Draw Rectangle and display the distance over Screen
 
    :param1 Image(Mat): simply the frame 
    :param2 Call_Out(bool): If want show Distance and Rectangle on the Screen or not
    :param3 Distance_Level(int): which change the line according the Distance changes(Intractivate)
    :return1  face_width(int): it is width of face in the frame which allow us to calculate the distance and find focal length
    :return2 face(list): length of face and (face paramters)
    :return3 face_center_x: face centroid_x coordinate(x)
    :return4 face_center_y: face centroid_y coordinate(y)
 
    '''
 
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        line_thickness = 2
        # print(len(faces))
        LLV = int(h*0.12)
        # print(LLV)
 
        
        cv2.line(image, (x, y+LLV), (x+w, y+LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y+h), (x+w, y+h), (GREEN), line_thickness)
        cv2.line(image, (x, y+LLV), (x, y+LLV+LLV), (GREEN), line_thickness)
        cv2.line(image, (x+w, y+LLV), (x+w, y+LLV+LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y+h), (x, y+h-LLV), (GREEN), line_thickness)
        cv2.line(image, (x+w, y+h), (x+w, y+h-LLV), (GREEN), line_thickness)
#         cv2.rectangle(image, (x, y), (x+w, y+h), BLACK, 1)
 
        face_width = w
        face_center = []
        # Drwaing circle at the center of the face
        face_center_x = int(w/2)+x
        face_center_y = int(h/2)+y
        if Distance_level < 10:
            Distance_level = 10
 
        # cv2.circle(image, (face_center_x, face_center_y),5, (255,0,255), 3 )
        if CallOut == True:
            # cv2.line(image, (x,y), (face_center_x,face_center_y ), (155,155,155),1)
            cv2.line(image, (x, y-11), (x+180, y-11), (ORANGE), 28)
            cv2.line(image, (x, y-11), (x+180, y-11), (YELLOW), 20)
            cv2.line(image, (x, y-11), (x+Distance_level, y-11), (GREEN), 18)
 
            # cv2.circle(image, (face_center_x, face_center_y),2, (255,0,255), 1 )
            # cv2.circle(image, (x, y),2, (255,0,255), 1 )
 
        # face_x = x
        # face_y = y
 
    return face_width, faces, face_center_x, face_center_y
```

### __Truy cập camera usb__


```python
# Camera Object
import cv2
import random
from PIL import Image
cap = cv2.VideoCapture(0)  # Number According to your Camera
Distance_level = 0
 
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
 
# face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```

### __Xử ý ảnh gốc__

Hãy chụp một bức ảnh của bạn để làm mẫu:
- Chụp lại khoảng cách cho phép trước.
- Độ rộng khuôn mặt của bạn.
Ví dụ:

Distance from camera to object(face) measured

___Known_distance = 40.2(cm);___

Mine is 14.3 something, measure your face width, are google it

___Known_width = 14.3(cm);___ 


![](https://lh3.googleusercontent.com/_-eTAX56q4mvdh0rrfbHfiB9rDHLK60xjm0fvjqYMaq5A7ArOXqMChnnCJ2R_4Ae7lIENO9LUZ45kRYwlItGv1vXhdOft77KqajTYhnGMrMSKNFASmqNgc1ulAvAC3x1_2O_zWsyC_9VBNQHB-869GQ9YqUICBcp0DfCrGGIh8JQRUYmVkGMTeELvGrsi-HhX3UYkZD2QSsRo6NQe3P6SN3ZOoUIIOaWdDNHumaSy7cXLPtye_dfcvGynoIcSD07HeayPNzg-UvnmKq5Ua0vmv97DJbplOAO1vY7We1FYSxHDC0opn0srKGWiOZujARNGMtOc2NbaBctNHPjy3r1_AipiAoTsLysh1MBgLGGPVoGyUDAvpPl7jHSsJgEzgiRLMiProxF-4i3ZATzFQWC0nHUclw4RKeHDjJclZtoZe1EdsdOI6U6uNbVmwrieO6oloNB7LI8v1IInUKu0L0fKRpmv-YIIaePAgoRsBdYYtgUq1AK8FHJUk2Zp3UAJbFM0u1J_UcIODfL_peK9xWSeHlCWP052AOyEJwlCOuU1-oG4d_kwvvqAWhtgHjmbGmGLTXMWrfRMvHTLlZr0T-hsEsRzOM0s8foENv2ZNY9DxrYQav_gFgh41qeOWPY5admpYqJnS5J0dCylf2Naiwgxxvf1fKKiLNoFJYgftUes2SiTfv5W0ezusciR9FV9l-pPOZl_1NPSkSdxLP3AvJkWkIi=w215-h286-no?authuser=0)




```python
image = Image.open('lam.jpg')
im = cv2.imread('lam.jpg')
print("Size webcam:  {}".format(im.shape))
new_image = image.resize((640, 480))
new_image.save('sdlam.png')
ref_image = cv2.imread("sdlam.png")
print("Size fix:  {}".format(ref_image.shape))
```

    Size webcam:  (480, 640, 3)
    Size fix:  (480, 640, 3)



```python
# reading reference image from directory 
ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(
    Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)
 
cv2.imshow("ref_image", ref_image)
 
while True:
    _, frame = cap.read()
    # calling face_data function
    # Distance_leve =0
 
    face_width_in_frame, Faces, FC_X, FC_Y = face_data(
        frame, True, Distance_level)
    # finding the distance by calling function Distance finder
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
 
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            # Drwaing Text on the screen
            Distance_level = int(Distance)
 
            cv2.putText(frame, f"Distance {Distance} cm",
                        (face_x-6, face_y-6), fonts, 0.5, (BLACK), 2)
    cv2.imshow("frame", frame)
#     out.write(frame)
 
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
# out.release()
cv2.destroyAllWindows()
```

    736.5314685314686


# __3.Kết quả:__

- Sử dụng OS:Ubuntu 20.04.02.
- Hình bên trái là nhận diện khuôn mặt.
- Hình bên phải là kiểm tra khoảng cách.
- Dùng camera USB để thực hiện. Độ phân giải 640x480.
Hiện tại sử dụng nhận diện khuôn mặt băng ___"CascadeClassifier"___. 

![](https://lh3.googleusercontent.com/XyvTgdPk2crI8DPh9xvH9XVZF60KooCChN2Q3CIMxDnjprlxJtuyib_-XCsatVMoIULMMIGeT8hC5OmSaixrJmK0pCBVm0yxh0-tC43La4yjrRkWg_fs-A6R5JP-zNHsIrK96ACRhEFTnEiuOeJLhgRt7k89l5NNikOI3NydFvwPbcnBhxvPh2f5_Vy3GtBSbvNvAz8AapK3v8w2it_pSEXjSjEDJn2LCze11lxBu-PvxD-De9BbY54LoGroJuaMlU1IggkiSkZNiX5PlO6G1EuOnHAsaJ7_xQ6Kzf5gIhuuXMVwiWlPZYUdHtK4xJ-rhf7muBJF8MgCEgaOCkjuRugUIe3a4gKopiBvdWSOQ7B7htnqta-G8nHPKFAFGo8RQi26U0Pb2Eyw6H1mGfLWQd7ZAGG2pae5NDXhU1gHRAgEA9xkgtOsm8fzyKJBPl16m6PsZOhQyZPKeLLErcnkCIkDyLZHccJijZ5JEr4Z-cgK0XMx5lbrEStp1iOKzjA2r86zFqi6WV8xc9YDNpH_UDtbE-O7fHCLdUyR6Ir3SJapCU5JYgaIDIq1vTUFoVH8fliE7WpWrHRddzwjBkFQtvL2Q4r0Byfix7i4G1xaQ0dTDJZNz20ZRJ_tj1sJCE_dIRQU2fXgT_4vga-5B8Z--u_j-g-OoP3Li2FabZd0AyxLwVNsvBckXAD6eOLVtiV0LduZPJ2wujfVMXxikeUzjKH0=w509-h286-no?authuser=0)

# __4. Tài liệu tham khảo__
http://ngoton.it/tinh-khoang-cach-tu-camera-den-mot-doi-tuong-voi-opencv/

https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

https://circuitdigest.com/tutorial/real-life-object-detection-using-opencv-python-detecting-objects-in-live-video
