# python_yolov4_custom_training

### 1. package download 
1) Darknet download
<p>미리 다운만 받아둔다. 파일 전처리 끝나고 사용할것임</p>
'''
git clone https://github.com/AlexeyAB/darknet.git
'''
2) OIDv4 Toolkit download 
'''
git clone https://github.com/EscVM/OIDv4_ToolKit.git
'''
<p>다운로드 후 해당 디렉토리로 이동 </p>
'''
cd OIDv4_ToolKit
'''
<p>install requirement.txt if need </p>
'''
pip3 install -r requirements.txt
'''

### 2. data download and pre processing 
1) Open Images Dataset download(Open Images Dataset V6 사용)

#python main.py downloader -y --classes [클래스이름] [클래스이름] --type_csv train --limit 1000 --multiclasses 1  
#limit 에 필요한 데이터 수량 지정
#한 폴더에 여러 클래스데이터를 받기위해 multiclasses 1 로 지정
#Example
python main.py downloader -y --classes Apple Banana Orange --type_csv train --limit 1000 --multiclasses 1

다운이 완료되면 OID/csv_folder 안에 csv 파일 2개, OID/Dataset 안에 jpg 데이터들과 label 폴더가 생긴지 확인 

2) classes.txt 파일을 copy 후 OID/Dataset/train/폴더이름 에 저장후, copy된 파일을 열어 class 이름을 업데이트  
'''
cp classes.txt /home/erin/Documents/project/OIDv4_ToolKit/OID/Dataset/train/Apple_Banana_Orange/classes.txt
'''

### 3. YOLOv4 (Darknet repository) 를 이용한 데이터 train
1) darknet 폴더로 이동 
'''
cd ..
cd darknet
''' 
2) Makefile 열어서 설정 : GPU=1,CUDNN=1, OPENCV=1  (GPU와 CUDNN이 깔려있지 않은경우 OPENCV만 변경해준다). 

3) make 명령어로 darknet 컴파일링
<p>compile darknet framework in order to use related files for training object detection model
make</p>

아래 명령어를 실행하여 usage: ./darknet <function> 가 나오면 설치가 제대로 된것
'''
./darknet 
'''
 
4) Pretrained weight 다운로드 
'''
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
'''
custom 파일 train 시
'''
#wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
'''

5) OIDTookit 에서 만들어둔 데이터를 copy 하여 darknet 하위 폴더에 저장
darknet/data 위치 폴더를 통째 복사 
'''
cp -r /home/erin/Documents/project/OIDv4_ToolKit/OID/Dataset/train/Apple_Banana_Orange /home/erin/Documents/project/darknet/data
'''

    

5) 만들어진 파일 실행, 제대 되었다면 OID/Dataset/train/폴더명 아래에 파일이 생성된걸 확인할수있음
* train.txt, test.txt, classes.name, image_data.data
'''
python generate_train.py
'''

6) cfg 파일 업데이트 (/darknet/cfg/yolov4.cfg)
* yolov4.cfg 파일을 copy 하여 아까 copy 해온 데이터셋 폴더아래 저장한다 (yolov4_custom.cfg)
'''
cp /home/erin/Documents/project/darknet/cfg/yolov4.cfg /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg
'''


6) 복사해온 파일을 열어 내용 정정 
* [net]
- batch= 기본값은 64
- subdivision= 배치사이즈 64를 얼마나 쪼개서 쓸것인지, 기본값은 8 (16,32,64 로 조정하여 사용가능)
- weight/height = 416으로 조정 (default 608 로하면 gpu 메모리를 많이 잡아먹음)
- max_batches = 2000 * num of classes (하나일경우 6000으로 지정)
- steps = 80%,90% of Max batches

* [yolo] : 총 3개가 있다**

- classes = 자기가만든 클래스 갯수
- 바로위 [convolutional]의 filter 업데이트 : Filter = (number of classes + 5) * 3   (공식에 관계없이 클래스가 하나면 filter=18, 4개면 filter=27 로 지정)

* 기타 추가항목에 대해서는 블로그 참조 [https://eehoeskrap.tistory.com/370](https://eehoeskrap.tistory.com/370)


7) 훈련실행 
* 만들어놓은 .data의 경로와 .cfg 경로를 적어주고 다운받은 weights를 넣어줌
'''
./darknet detector train /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/image_data.data /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg yolov4.conv.137 -dont_show -map
'''

8) 훈련을 마치면 만들어진 weights 파일을 가지고 테스트 (/backup)
'''
./darknet detector test /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/image_data.data /home/erin/Documents/project/darknet/data/Apple_Banana_Orange/yolov4_custom.cfg /home/erin/Documents/project/darknet/backup/yolov4_custom_last.weights /home/erin/Downloads/1234.jpg
'''
