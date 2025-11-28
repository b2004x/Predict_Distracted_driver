# Application of models to detect distracted driving behavior of drivers

## Description
- **The project  “Application of models to detect distracted driving behavior of drivers” will focus on building a system using computer vision and artificial intelligence to automatically identify dangerous driving behaviors of drivers while driving and delivers safety alerts to help prevent accidents.**
-**This model require the camera angel that can cover the whole the whole upper body and the steering wheel**

## Features
- **Predict the driver behavior while driving.**
- **Give out alerts when the driver is being distracted.**

## Sample 
![alt text](/Vid_Demo/demo1.PNG)

![alt text](/Vid_Demo/safe_driving.PNG)

![alt text](/Vid_Demo/texting.PNG)

![alt text](/Vid_Demo/other_activities.PNG)

![alt text](/Vid_Demo/safe_driving.PNG)

## Dataset
-**This model use the the distracted driver dataset.**
-**This model require the camera angel that can cover the whole the whole upper body**
**Dataset: https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data**
```
data/
└── driver_behavior/
    ├── train/
    │   ├── other_activities/
    │   │   ├── 2019-04-2416-05-13.png
    │   │   ├── 2019-04-2416-06-20.png
    │   │   └── 2019-04-2416-06-3.png
    │   │   └── ...
    │   ├── safe_driving/
    │   ├── talking_phone/
    │   ├── texting_phone/
    │   └── turning/
    └── val/
        ├── other_activities/
        ├── safe_driving/
        ├── talking_phone/
        ├── texting_phone/
        └── turning/
```

## Model 
**Choose the model that achives the best results.**

- **VGG19, a deep convolutional neural network with 19 layers that uses simple stacked 3×3 convolutions to achieve strong image classification performance.**

-**ResNet50, a 50-layer deep neural network that uses residual skip connections to enable very deep training without vanishing gradients, achieving strong performance in image recognition tasks.**

-**EfficientNet-B0 is* a compact, highly efficient CNN that uses compound scaling of depth, width, and resolution to achieve strong accuracy with significantly fewer parameters and FLOPs.**

## Model performance
### Resnet50

![alt text](/Vid_Demo/cm_resnet50.PNG)

![alt text](/Vid_Demo/cr_Resnet50.PNG)

![alt text](/Vid_Demo/ac_resnet50.PNG)

-**The model perform very well on this dataset, most of the class have more than 90% accuracy except for the other_activites class.**
-**The model still sometime predict the wrong class as seen in the confusion matrix, epescially texting_phone and turning.**
-**The model start to converging in epoch 6 or 7 as seen in the diagram.**

### EfcientNet_B0 

![alt text](/Vid_Demo/cr_efficientNet.PNG)

![alt text](/Vid_Demo/cm_efficientNet.PNG)

![alt text](/Vid_Demo/ac_efficientNet.PNG)

-**The model also perform very well on this dataset, every class have more than 90% accuracy, even texting_phone have 100% accuracy.**
-**Only a small number of images been classify wrong.**
-**The train loss and accuracy line is more stable than the validation line, but the model still perform very well and have high accuracy and very low loss.**
-**At first this maybe the case of overfitting but the model still demonstrates strong performance on the validation set.**

### VGG19


![alt text](/Vid_Demo/cr_VGG19.PNG)

![alt text](/Vid_Demo/cm_VGG19.PNG)

![alt text](/Vid_Demo/ac_VGG19.PNG)

-**The model perform very good on this dataset with the 97% accuracy on the train set and 95% accuracy on the validation set.**
-**There are still some image been wrongly classify.**
-**The model converging very soon at epoch 4 or 5.**


![alt text](/Vid_Demo/Comparision_1.PNG)

![alt text](/Vid_Demo/Comparision_2.PNG)
-**Choose EffcientNet_B0 for is outstanding accuracy, low train/validation loss and very low trainning parameter.**
-**This is the most effective model out of the three model that we test.**
## Instruction
-**In the working folder, open terminal.**

```
streamlit run app.py
```


### Demo 


![alt text](/Vid_Demo/demo1.PNG)

![alt text](/Vid_Demo/other_activities.PNG)

![alt text](/Vid_Demo/safe_driving.PNG)

## Referances

-**3. ·. H. ·. A. ·. M. I. Md. Ashraf Uddin1, " AbnormalDriving Behavior Detection," 2025.**

-**VNEXPRESS, "Tai nạn giao thông giảm trong 5 ngày nghỉ lễ," VNEXPRESS, 04 05 2025. [Online]. Available: https://vnexpress.net/tai-nan-giao-thong-giam-trong-5-ngay-nghi-le-4881467.html.**

-**W. P. Review, "Countries with the Most Car Accidents 2025," World Population Review, [Online]. Available: https://worldpopulationreview.com/country-rankings/countries-with-the-most-car-accidents.**

-**aicandy, "Mô hình ResNet - Đột phá trong nhận diện hình ảnh," 2024.**

-**M. T. Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," 2020.** 

-**A. R. J. FAIQA SAJID1, "Distracted Driver Detection," 31 12 2021.**

-**. M. I. K. K. P. NAVEEN KUMAR VAEGAE, "Design of an Efficient Distracted Driver Detection," 2022.**


