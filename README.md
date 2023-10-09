# Cloud Classification For Weather Forecast

# :speech_balloon: What is "Cloude Classification For Weather Forecast"?

1. 본 프로젝트는 기상예보의 실시간 정확도를 좀더 높히고자 하는 목적을 가지고 있다.
2. 현재 초단기 기상예보의 경우 '기상 예보관' 이라는 인적 요인을 필요로 하며 실시간으로 관측을 하는 역할이다.
3. 예보관의 주관에만 의존하기 보다 AI모델을 적용시켜 예보관을 도와주고자 하여 만든 프로젝트이다. 

## :date:Project Planning

- **기간** : 2023.04.01 ~ 2023.04.21 (약 3주)
- **인원 구성** : 3명

- ** :boy: My part**
    - Adaptive Binarization을 적용시킨 **Custom Dataset 생성**
    - **Dataset Annotation** 및 Gray-Scaling, Histogram Equalization, Zero-Centering를 적용하여 **데이터 전처리 작업**
    - 모델 복잡도에 따른 **모델 선정** 및 Anchor Box, max batch 조정을 통한 Yolo **모델 핸들링**
- **사용언어 및 개발환경** : Google colab Pro+, Python, Numpy, Pandas, Matplotlib, OpenCV

##  :boom:문제 정의
### 기상청 정확도가 90%라고?
---
<img width="1044" alt="스크린샷 2023-07-27 오후 1 23 15" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/f9356461-9c71-49fe-af15-af9ff966d756">

- 기상청은 현재 수치형 Table 모델을 가지고 날씨를 예측 하고 있고 이 정확도는 무려 90%이다.
- 하지만 시민들이 느끼기에는 90%라기에는 갑작스러운 기상변화를 한번쯤은 느꼈을 것이다
- 그 이유는 기상을 예측이 일반 사람이 생각하는 것보다 세분화 되어있다는 것이다.
- 날씨를 예측하는데 있어서 장기, 단기, 초단기와 같이 세분화 되어있다.
- 90%의 정확도는 장기예보에 해당하는 정확도이고 초단기 예보 (약 10분 이내)는 높은 정확도를 보여주지 않는다.


<img width="1037" alt="스크린샷 2023-07-27 오후 1 23 37" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/ed569f82-0608-4fde-8355-05f01bffc81a">

- 초단기 일기예보의 경우 실시간으로 변하는 요인들이 너무많기에 AI의 힘만으로는 예측이 어렵다.
- 따라서 예보관이라는 전문가가 필요하게 되고 예보관이 하는일은 [**그래프 분석**, **구름이 특정 모양이 되었을때 바로 포착해야 하는 것**]과 같은 작업들이 필요하다.


## How to solve this problem?

### <Object detection 방법>

- 객체탐지라는 기술을 통하여 실시간으로 위성위에서 움직이는 구름을 포착하여 예보관이 실시간으로 체크를 하지 않도록 해준다
- 인간이 분별하기 힘든 여러 구름을 다양한 전처리를 통해서 정확도를 높히는 방법을 이용한다
- 수만장의 위성사진을 확보하여 직접 라벨링 작업을 하여 학습을 진행한다.


### <무엇을 얻을 수 있나?>

- AI 인공지능 수치 모델 이외에 딥러닝 서비스를 제공하기 위하여 인공위성 사진으로부터 날씨와 연관된 구름의 형태를 분석하여 초단기 일기 예보관에게 보조 수단으로서 기상에 대한 분석 정보를 제공


## 데이터 설명

- **데이터** : NASA WORLD VIEW 35,050장
    
    <img width="943" alt="스크린샷 2023-07-27 오후 1 24 14" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/ccf49809-2c33-4540-9cb5-02bac1cd55cb">
    
    - image size: 2100 x 1400

    - Label Type: Sugar, Gravel, Fish, Flower

    - Noise: 인공위성 지지대, 태양빛 반사 

<br>

    
- **Label 분포 그래프**

    <img width="618" alt="스크린샷 2023-07-27 오후 1 24 35" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/bde033b1-4d17-4629-a9ba-2d8c0280a5e7">

<br>

- **날씨 예보와 관련된 구름 규모**

    <img width="621" alt="스크린샷 2023-07-27 오후 1 25 02" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/9fc99e32-69c7-4534-8e5c-f97b34c64e55">

    - 구름의 규모는 총 4 구간으로 나뉘며, 규모 중에서도 날씨와 가장 연관성이 큰 **Meso-manufacturing** 구간을 기준으로 하여 4개의 Class를 분류한다.


    - **Sugar[meso-γ]** : 강수와 연관성이 거의 없고, 설탕을 뿌린 모양과 유사한 Sugar 구름은 여러 입자로 이루어진 미세 구름이며, 낮은 고도에서 관측된다.
    - **Gravel[meso-β]** : 주로 대기의 불안정에 의해 발생하기에 강수와 연관성이 있고, 돌풍으로 인하여 Sugar 구름에서 파생된 Gravel 구름은 Sugar 구름에 비해 입도가 크고 밝다.
    - **Flower[meso-β]** : 안정적인 기상 조건에서 발생하며, 규칙적으로 넓게 퍼져 있는 패턴의 Flower 구름은 Gravel 구름이 합쳐진 형태이다.
    - **Fish[meso-α]** : 갑작스러운 폭우와 비가 있을 때의 형상으로, 구름 사이에 뼈대와 같은 모양이 관측된다. 또한, Sugar와 Gravel보다 밝은 Fish 구름은 태풍으로의 발전 가능성이 존재하며, 열대 저기압이 발생하는 곳에서 많이 관측된다.

<img width="1008" alt="스크린샷 2023-07-27 오후 1 25 55" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/37d40ade-e5ea-497b-886d-4dd650b99fd5">

- **Dataset의 Bbox와 Class**
    <img width="1008" alt="스크린샷 2023-07-27 오후 1 26 19" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/226fea58-d68c-4ada-8584-a5966d8afc14">    

## 데이터 전처리 방법

- **HSV → ‘A’SV**
    
    <img width="1008" alt="스크린샷 2023-07-27 오후 1 26 42" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/8bc09a49-70a9-47c3-9814-6db680d14b5d">


    - 색공간 모델 중 세 가지 요소로 색을 표현하는 방식이며, 색상(Hue)은 [0, 255]의 값으로 색 종류를, 채도(Saturation)는 [0, 255]의 값으로 색의 밀도를, 명도(Value)는 [0, 255]의 값으로 색의 밝기를 표현한다.
    - 두 번째 그래프는 한 시점에서 시간이 흐름에 따라 달라지는 구름의 높이를 측정한 그래프를 나타낸다.
    - **색상**(Hue)은 비교적 비슷하여 Class 별로 비교하기 어렵지만, **채도**(Saturation)는 구름의 밀도 정보를 반영하고 있어 구름의 높이가 높을수록 채도가 높다는 것을 알 수 있으며, 이러한 기준으로 쌓여있는 구름이라고 표현할 수 있다.
    - **명도**(Value)는 구름의 밝기 정보를 반영하고 있어 인공위성 촬영 시점에서 바라봤을 때 구름의 평면이 넓을수록 명도가 높다는 것을 알 수 있으며, 이러한 기준으로 손을 잡고 있는 구름이라고 표현할 수 있다.

  <br>


    <img width="1008" alt="스크린샷 2023-07-27 오후 1 27 07" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/76260c11-1633-4de1-b747-b76d90b93f08">

    - **Noise Image** : Dataset에는 인공위성 지지대, 태양빛과 같은 Noise가 존재하는 Image
    - **Adaptive Binarization Image** : 이미지의 각 픽셀을 이진화할 때 주변 픽셀의 평균값을 기준으로 0 또는 1으로 변환한 Image

    <br>
  
    <img width="1018" alt="스크린샷 2023-07-27 오후 1 27 59" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/8523f8fe-1ebc-4133-955e-27c5fab0281d">


    
    - 기존의 색상(Hue) Image는 패턴 파악에 악영향을 주기 때문에 패턴 학습의 난이도를 낮추기 위해 Class 별로 비교하기 어려운 색상(Hue) Image를 기존의 Image에 Adaptive Mean Binarization을 적용한 **Adaptive Binarization Image**로 대체하여 이미지의 광도 불균일성이나 조명 변화와 같은 영향을 최소화하여 모델의 성능을 개선하였다.

    <br>


- **이외의 전처리[Gray-Scaling, Histogram Equalization, Zero-Centering]**


    <img width="1007" alt="스크린샷 2023-07-27 오후 1 28 29" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/0a23236d-b930-418a-bfe6-70451808c3aa">

    - **Histrogram Equalization Image** : 해당 이미지의 픽셀 값 빈도 분포를 균등하게 만드는데, 상대적으로 밝은 부분은 덜 밝은 부분보다 더 밝게, 덜 밝은 부분은 더 어둡게 만드는 것으로, 이미지의 대비를 개선해 시각적 품질을 향상시킨 Image
    - Adaptive Binary 이외에도 다양하게 전처리를 적용해 주었지만, 상대적으로 모델의 성능이 개선되지 않았다.

<br>

- **Dataset Labeling**

    <img width="1006" alt="스크린샷 2023-07-27 오후 1 29 03" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/ceb8c3f4-1a6f-40df-a6d3-b99628acd71b">

    - YOLO v8 모델의 성능 자체가 좋지 않았으며, Roboflow에서 제공되고 있는 Dataset을 확인한 결과 왼쪽의 두 사진과 같이 Annotation에 문제가 있었기에 모든 사진을 확인하면서 잘못된 Annotation Bounding Box와 중복된 Annotation Bounding Box를 제거하고, Bounding Box가 중복되지 않도록 **Annotation을 조정**하여 모델의 성능을 개선하였다.

## 모델 성능 개선

- **YOLO v4 [Anchor Box 조정]**
    
    <img width="1273" alt="스크린샷 2023-07-27 오후 1 29 47" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/b0cddace-523b-4030-8c9a-80c96dcaab86">

    - **Bounding Box vs. Anchor Box** : Bounding Box는 실제 객체의 위치와 크기를 나타내는 박스이며, Anchor Box는 모델이 객체를 검출하는 시작점으로서, 이미지에서 객체가 있을 가능성이 높은 위치와 크기가 사전에 정의된 박스를 말한다.
    - Cloud Dataset Image의 Bounding Box Scale을 나타내는 왼쪽의 그래프는 한 곳에 밀집되어 있는 Yolo v4의 Default Anchor Box Scale과 맞지 않아 객체 인식을 잘 하지 못 하는 문제가 발생하였다.
    - 이를 해결하기 위해 K_means로 총 9개의 Cluster를 생성하였으며, 생성된 Cluster의 중심 좌표를 Anchor Box의 좌표로 이용하여 **Yolo v4의 Anchor Box의 분포를 조정**해 모델의 성능을 개선하였다.
    
    <img width="689" alt="스크린샷 2023-07-27 오후 1 30 47" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/c05ddda5-a543-460f-9fce-9af740bc5669">

    - **mAP** : 4개의 Class에 대하여 각각 Precision(모델이 True라고 예측한 것중에 맞춘 Score)-Recall(실제로 True인 것중에 모델이 맞춘 Score) 곡선 아래 면적의 평균을 구하고, 모든 Class의 구한 값들인 Average Precision을 평균낸 지표
    - Anchor Box 조정 전에는 mAP Score가 0.306이었지만, 조정 후의 mAP Score는 0.335로 0.035 상승함에 따라 Anchor Box를 조정하여 모델의 성능이 개선된 것을 볼 수 있다.

- **YOLO v5 [ASV 적용]**
    <img width="701" alt="스크린샷 2023-07-27 오후 1 32 02" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/1f229425-55a4-442c-b3f3-512ffcbce792">

    - HSV는 빛에 대한 정보 손실이 컸기에 색상(Hue) Image를 Adaptive Binarization Image로 대체하여 태양빛 Noise를 제거한 결과 YOLO v5의 HSV 데이터셋 mAP Score는 0.353이지만, ASV 데이터셋 mAP Score는 0.358으로 0.005 상승한 것을 확인할 수 있다.
    - 다른 구름에 비해 상대적으로 흐릿하게 보여지는 Sugar 구름은 태양빛 Noise에 악영향을 받기 때문에 Adaptive Binarization Image를 통해 Sugar 구름의 AP Score를 0.09 상승시켜 Sugar 객체 인식 성능을 개선하였다.
    
    <img width="694" alt="스크린샷 2023-07-27 오후 1 32 28" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/ebc713fa-f088-45aa-a456-3e591cada1db">

    - ASV를 통한 성능 향상을 기대했지만, 원본 이미지의 Score가 0.366으로 0.008 더 높게 나왔다.

- **YOLO v8[모델 복잡도]**
    
    - **모델 복잡도에 따른 성능 차이**

        <img width="660" alt="스크린샷 2023-07-27 오후 1 33 11" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/b479cf39-1a54-4fce-be11-a7e6f7064e69">
    
        - YOLO v8 모델은 파라미터의 개수에 따라 5개의 Scale으로 나뉘어져 있어 모델의 Scale을 선택할 수 있었으며, 모델의 복잡도가 상대적으로 높은 v8l 모델이나 v8x 모델이 v8s 모델, v8m 모델보다 성능이 좋지 않았다.
        - 이에 따라 데이터와 모델의 복잡도에 따라 성능 차이가 존재한다는 것을 알 수 있었으며, 이를 통해 최종적으로 **YOLO v8s 모델**을 사용하였다.
    - **YOLO v8s 모델의 mAP 그래프**

        <img width="431" alt="스크린샷 2023-07-27 오후 1 33 41" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/ed18a689-6856-4762-98e8-7b790baca3a0">

        - YOLO v8 모델의 Anchor Box는 Auto Anchor 방식이며, YOLO v4나 YOLO v5보다 상대적으로 모델 복잡도가 높기에 Class 분류가 쉬워져 RGB 데이터셋의 YOLO v5 mAP Score인 0.366에 비해 0.03 상승한 것을 볼 수 있다.
        - YOLO v8의 인프라는 사용하기 수월했지만, 23.04.19 기준으로 논문이 발표되지 않아 구조 파악이 난해하였다.

- **YOLO v8 [Annotation 조정]**

    <img width="658" alt="스크린샷 2023-07-27 오후 1 34 19" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/fa4b80fc-18ca-490b-a15e-35e7d19591aa">

    - 모든 Image의 Annotation을 조정해 주었을 때 mAP Score는 크게 증가하지 않았지만, F1-Score 그래프를 통해 각 Class의 Confidence가 상승하면서 mAP의 면적이 증가하는 것을 볼 수 있었다.
    - 또한, 전체적으로 모델의 Score 중에서 Flower 구름의 Score는 높았지만, Sugar 구름과 Gravel 구름, Fish 구름이 상대적으로 낮은 Score로 측정 되었다.
    - 이는, 구름 패턴의 차이를 명확히 하지 않은 상태로 Annotation 되어 mAP가 크게 증가하지 않았던 것으로, 더 확실한 annotation이 필요하다고 생각한다.

## 개선사항 & 기대효과 & Inference

- **개선사항**
    - Dataset Annotation 시 명확한 기준으로 확실한 Annotation이 되어야 한다.
    - 영역이나 패턴 학습에 특화된 Segmentation 모델을 사용한다면 모델의 성능을 더 개선할 수 있을 것이라 생각된다.
    - Coco Pretrained Model을 사용하였는데, 구름 패턴 학습에 적합한 Pretrained Model을 찾는다면 성능 개선의 여지가 있을 것이라 생각된다.

- **기대효과**
      <img width="537" alt="스크린샷 2023-07-27 오후 1 34 36" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/1e244062-1ed6-49ac-a8e3-03cb20dba1c1">

    
    - 초단기 예보관에게 정확한 시각적 정보를 빠르게 제공함으로서 예보관의 초단기 날씨 예측 의사결정에 도움을 줄 수 있다.
    - 머신용 데이터 제공이 가능하다.
    - 일반인들이 위성 사진을 보았을 때 시각적으로 강수 확률 확인이 가능하다.

- **Inference**
    - **모델 복잡도가 상대적으로 낮은 v8s 모델과 v8m 모델의 Inference 결과**
      <img width="534" alt="스크린샷 2023-07-27 오후 1 34 50" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/b85b65b3-dace-4f45-a98e-113556e05e24">

    
    - **2023년 04월 20일 07:30 ~ 08:30 한반도 위성 영상**
![구름](https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/17322f86-4efa-42da-8697-21af4e63e5f3)

    - **Inference 결과**
        <img width="458" alt="스크린샷 2023-07-27 오후 1 35 42" src="https://github.com/Yu-Miri/Cloud_Classification_for_Weather_Forecast/assets/121469490/5eb7598a-af80-45df-acd4-7a34c31fec05">
        
        - 실제 4월 20일 북한 및 중국 지역에 Fish 구름이 포착되었고 실제 이 지역에는 비가 오고 있었다. 

## Installation

### Requirements
~~~
git clone https://github.com/Yu-Miri/Cloud_Classification_For_Weather_Forecast.git

# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
cd tensorflow-yolo4-tflite
pip install -r requirements.txt

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
cd tensorflow-yolo4-tflite
pip install -r requirements-gpu.txt
pip install ultralytics
~~~

### Preparing the Dataset
https://drive.google.com/drive/folders/1MlQITILnZcO-A8KWyLbuo1QYInxkkMap?usp=sharing

### Training

~~~
cd Cloud_Classification_For_Weather_Forecast
./darknet detector train data/cloud_data.data cfg/yolov4-custom.cfg /content/drive/MyDrive/object/darknet/yolov4.conv.137 -dont_show -map
~~~

### Inference
- Yolo v4 Inference
from inference import yolov4_inference
yolov4_inference(img, './cfg/yolov4-custom_test.cfg', './data/cloud_data.data', './data/yolov4-custom_60.weights') 
~~~

- Yolo v8 Inference
~~~
from img_preprocess import resize_img
path = {img_file}
img_resize = resize_img(path)
path = path[:-4] + '_640.png'
cv2.imwrite(path, img_resize)
!yolo task=detect mode=predict model='./pth/yolov8s_best.pt' conf=0.189 source={path}, imgsz= 640

result_path = './detect/predict/' + path.split('/')[-1]
img = cv2.imread(result_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
~~~
