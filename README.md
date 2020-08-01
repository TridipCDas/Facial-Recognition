# Facial-Recognition

Build a face recognition system comprising the following steps:
1. First perform face detection. 
2. Extract face embeddings from each face using deep learning. 
3. Train a face recognition model on the embeddings.
4. Recognize faces in both images and video streams with OpenCV.

#### HOW TO USE IT:

1. Download the  Torch deep learning model which produces the 128-D facial embeddings.
   Download link :https://drive.google.com/file/d/1NloyNmJDWPKqX4_h2W_rwssPrERma354/view?usp=sharing
   Paper : https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf
   
2. Inside the dataset folder, keep the images of person organized into subfolders by name. For
   this repository, I have used My images and the images of our Honourable Prime Minister Narendra Modi.
   
3. After that, run the file extract_embeddings.py to generate 128 D facial embeddings of our images.

4. Run train_model.py to train on these embeddings.

5. Then at last run either recognize_image.py or recognize_video.py as per use.


#### OUTPUT:

![Screenshot (349)](https://user-images.githubusercontent.com/40006730/89109405-7ac58200-d45e-11ea-9858-da292ef436e9.png)

![Screenshot (350)](https://user-images.githubusercontent.com/40006730/89109408-7e590900-d45e-11ea-80dc-3cb803843673.png)

#### REFERENCES: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
 
