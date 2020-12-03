# python_yolov4_custom_training


1. package download 
1) Darknet download
#미리 다운만 받아둔다. 파일 전처리 끝나고 사용할것임 
git clone https://github.com/AlexeyAB/darknet.git

2) OIDv4 Toolkit download 
git clone https://github.com/EscVM/OIDv4_ToolKit.git
#다운로드 후 해당 디렉토리로 이동 
cd OIDv4_ToolKit

#**install requirement.txt if need: pip3 install -r requirements.txt

2. data download and pre processing 
1) Open Images Dataset download(Open Images Dataset V6 사용)

#python main.py downloader -y --classes [클래스이름] [클래스이름] --type_csv train --limit 1000 --multiclasses 1  
#limit 에 필요한 데이터 수량 지정
#한 폴더에 여러 클래스데이터를 받기위해 multiclasses 1 로 지정
#Example
python main.py downloader -y --classes Apple Banana Orange --type_csv train --limit 1000 --multiclasses 1

다운이 완료되면 OID/csv_folder 안에 csv 파일 2개, OID/Dataset 안에 jpg 데이터들과 label 폴더가 생긴지 확인 

2) convert_annotation.py 파일 생성 및 저장 (경로: /OIDv4_Toolkit)
 
 #convert_annotation.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import fileinput

# function that turns XMin, YMin, XMax, YMax coordinates to normalized yolo format


def convert(filename_str, coords):
    os.chdir("..")
    image = cv2.imread(filename_str + ".jpg")
    coords[2] -= coords[0]
    coords[3] -= coords[1]
    x_diff = int(coords[2]/2)
    y_diff = int(coords[3]/2)
    coords[0] = coords[0]+x_diff
    coords[1] = coords[1]+y_diff
    coords[0] /= int(image.shape[1])
    coords[1] /= int(image.shape[0])
    coords[2] /= int(image.shape[1])
    coords[3] /= int(image.shape[0])
    os.chdir("Label")
    return coords


ROOT_DIR = os.getcwd()

# create dict to map class names to numbers for yolo
classes = {}
with open("classes.txt", "r") as myFile:
    for num, line in enumerate(myFile, 0):
        line = line.rstrip("\n")
        classes[line] = num
    myFile.close()
# step into dataset directory
os.chdir(os.path.join("OID", "Dataset"))
DIRS = os.listdir(os.getcwd())

# for all train, validation and test folders
for DIR in DIRS:
    if os.path.isdir(DIR):
        os.chdir(DIR)
        print("Currently in subdirectory:", DIR)

        CLASS_DIRS = os.listdir(os.getcwd())
        # for all class folders step into directory to change annotations
        for CLASS_DIR in CLASS_DIRS:
            if os.path.isdir(CLASS_DIR):
                os.chdir(CLASS_DIR)
                print("Converting annotations for class: ", CLASS_DIR)

                # Step into Label folder where annotations are generated
                os.chdir("Label")

                for filename in tqdm(os.listdir(os.getcwd())):
                    filename_str = str.split(filename, ".")[0]
                    if filename.endswith(".txt"):
                        annotations = []
                        with open(filename) as f:
                            for line in f:
                                for class_type in classes:
                                    line = line.replace(
                                        class_type, str(classes.get(class_type)))
                                labels = line.split()
                                coords = np.asarray([float(labels[1]), float(
                                    labels[2]), float(labels[3]), float(labels[4])])
                                coords = convert(filename_str, coords)
                                labels[1], labels[2], labels[3], labels[4] = coords[0], coords[1], coords[2], coords[3]
                                newline = str(labels[0]) + " " + str(labels[1]) + " " + str(
                                    labels[2]) + " " + str(labels[3]) + " " + str(labels[4])
                                line = line.replace(line, newline)
                                annotations.append(line)
                            f.close()
                        os.chdir("..")
                        with open(filename, "w") as outfile:
                            for line in annotations:
                                outfile.write(line)
                                outfile.write("\n")
                            outfile.close()
                        os.chdir("Label")
                os.chdir("..")
                os.chdir("..")
        os.chdir("..")
        

- convert_annotation.py 실행으로 label 안에 있던 .txt 데이터를 train에 필요한 형식에 맞게 처리해줌
- .jpg와 .txt 이름이 같아야함

3) classes.txt 파일을 copy 후 OID/Dataset/train/폴더이름 에 저장후, copy된 파일을 열어 class 이름을 업데이트  
cp classes.txt /home/erin/Documents/project/OIDv4_ToolKit/OID/Dataset/train/Apple_Banana_Orange/classes.txt

3. YOLOv4 (Darknet repository) 를 이용한 데이터 train
1) darknet 폴더로 이동 

cd ..
cd darknet

2) Makefile 열어서 설정 : GPU=1,CUDNN=1, OPENCV=1  (GPU와 CUDNN이 깔려있지 않은경우 OPENCV만 변경해준다). 

3) make 명령어로 darknet 컴파일링
#compile darknet framework in order to use related files for training object detection model
make

# 아래 명령어를 실행하여 usage: ./darknet <function> 가 나오면 설치가 제대로 된것
./darknet 
  
4) Pretrained weight 다운로드 
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
#custom 파일 train 시 (con
#wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

5) OIDTookit 에서 만들어둔 데이터를 copy 하여 darknet 하위 폴더에 저장
#darknet/data 위치 폴더를 통째 복사 
cp -r /home/erin/Documents/project/OIDv4_ToolKit/OID/Dataset/train/Apple_Banana_Orange /home/erin/Documents/project/darknet/data

6) generate_train.py 파일 생성 및 저장 (경로: /darknet)
touch generate_train.py


#generate_train.py
#train 데이터를 80%, test 데이를 20%로 지정함
import os

#set Image Directory path  => 각자의 path에 맞게 설정
image_path = '/home/erin/Documents/project/darknet/data/Apple_Banana_Orange'  
#copy annotated data and paste to the new folder 
os.chdir(image_path)

#Iterate through each image found in the directory and save the corresponding path to the list called as path_list
path_list=[]
#Go through all the image files in the directory 
#Fullstop in os.walk('.)means the current directory
for current_dir,dirs,files in os.walk('.'):
    #iterate through all the files
    for f in files:
        #check if the file extension ends with '.jpg'
        if f.endswith('.jpg'):
            #prepare file path to save into train.txt
            file_loc=image_path + '/' + f 
            #append the path data into list "path_list". New line character \n is used to write the new content 
            path_list.append(file_loc + '\n')

#Divide the data into 80:20 ratio. Get 20% of data from path_list
# to write into the test.txt file 
path_list_test = path_list[:int(len(path_list)*0.20)]
#delete the same 20% records from the path_list as that 20% data is in path_list_test now
path_list = path_list[int(len(path_list)*0.20):]


#create train.txt file and write 80% of data(lines) inside it
with open ('train.txt', 'w') as train:
    #Iterate through all the elements in the list
    for i in path_list:
        #write the current path at the end of the file
        train.write(i)

#create text.txt file and write 20% of data (lines) inside it
with open('test.txt','w') as test:
    #Iterate through all the elements in the list 
    for i in path_list_test:
        #write the current path at the end of the file
        test.write(i)


# Initialize the counter 
i=0
#create classes.names file by reading content from existing classes.txt file 
with open(image_path + '/'+'classes.names','w') as cls, open(image_path + '/'+ 'classes.txt','r') as text:
    #iterate through indivisual lines in classes.txt file and write them into classes.name file
    for l in text:
        cls.write(l)
        #increasing counter 
        i+=1

#create image_data.data 
with open(image_path + '/' +'image_data.data','w') as data:
    #write number of classes 
    data.write('classes = '+ str(i)+'\n')
    #write fully qualified path of the train.txt file
    data.write('train='+image_path + '/'+'train.txt'+'\n')
    #write fully qualified path of the train.txt file
    data.write('valid='+image_path + '/'+'test.txt'+'\n')
    #write fully qualified path of the classes.names file 
    data.write('names='+image_path+'/'+'classes.names'+ '\n')
    #specify folder path to save trained model weights
    data.write('backup = backup')
    
    
    

#만들어진 파일 실행, 제대 되었다면 OID/Dataset/train/폴더명 아래에 파일이 생성된걸 확인할수있음
#train.txt, test.txt, classes.name, image_data.data
python generate_train.py

6) cfg 파일 업데이트 (/darknet/cfg/yolov4.cfg)
#yolov4.cfg 파일을 copy 하여 아까 copy 해온 데이터셋 폴더아래 저장한다 (yolov4_custom.cfg)
cp /home/erin/Documents/project/darknet/cfg/yolov4.cfg /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg



복사해온 파일을 열어 내용 정정 
**[net]**

- batch= 기본값은 64
- subdivision= 배치사이즈 64를 얼마나 쪼개서 쓸것인지, 기본값은 8 (16,32,64 로 조정하여 사용가능)
- weight/height = 416으로 조정 (default 608 로하면 gpu 메모리를 많이 잡아먹음)
- max_batches = 2000 * num of classes (하나일경우 6000으로 지정)
- steps = 80%,90% of Max batches

**[yolo] : 총 3개가 있다**

- classes = 자기가만든 클래스 갯수
- 바로위 [convolutional]의 filter 업데이트 : Filter = (number of classes + 5) * 3   (공식에 관계없이 클래스가 하나면 filter=18, 4개면 filter=27 로 지정)

기타 추가항목에 대해서는 블로그 참조 [https://eehoeskrap.tistory.com/370](https://eehoeskrap.tistory.com/370)


7) 훈련실행 
#만들어놓은 .data의 경로와 .cfg 경로를 적어주고 다운받은 weights를 넣어줌
./darknet detector train /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/image_data.data /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg yolov4.conv.137 -dont_show -map

8) 훈련을 마치면 만들어진 weights 파일을 가지고 테스트 (/backup)

./darknet detector test /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/image_data.data /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg /home/erin/Documents/project/darknet/backup/yolov4_custom_last.weights /home/erin/Downloads/1234.jpg
