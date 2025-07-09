Title:
Speech Emotion Recognition using Deformable Convolutional Neural Networks (DCNN)

Overview:
This project focuses on building a robust Speech Emotion Recognition (SER) system using Deformable Convolutional Neural Networks (DCNN). The system classifies human emotions from speech signals by leveraging the flexibility of deformable convolutions, which adapt more effectively to variations in the input feature maps compared to traditional CNNs.

Motivation:
Emotion recognition from speech is crucial in making human-computer interactions more natural. Traditional CNNs often fall short when dealing with the irregular and variable nature of emotional speech. Deformable convolutions allow the network to adjust its receptive fields dynamically, making it better suited for capturing the subtle variations and non-linear patterns in emotional speech data.

Model Architecture:
The model uses a combination of Mel-Spectrogram and MFCC features stacked as a dual-channel input. A custom DCNN architecture is designed using TensorFlow and Keras. It includes a series of convolutional layers (including deformable ones), batch normalization, dropout, and dense layers with softmax activation for multi-class emotion classification.

Datasets Used:
The model was trained on a merged dataset composed of:
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
TESS (Toronto Emotional Speech Set)
CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
Only 8 core emotion classes were used to maintain balance and consistency.

Feature Extraction:
Audio files were converted into: MFCC (Mel-Frequency Cepstral Coefficients), Mel-Spectrogram. These features were normalized and padded to a fixed length, then stacked into a shape of [40, 173, 2] before feeding into the model.

Model Training:
Framework: TensorFlow and Keras
Loss Function: Categorical Crossentropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
Epochs: Tuned based on performance (usually between 50–100)
Data split: Stratified for training and testing

Web Application:
A simple Flask-based web interface is provided to allow users to upload .wav audio files and get emotion predictions in real time. The front-end is built using HTML, CSS, and JavaScript, designed to be clean and responsive.

Results:
The system achieved high accuracy and performance metrics on the test data, showing strong generalization across speakers and emotions. Deformable convolutions showed noticeable improvements over standard CNNs in handling variable-length and speaker-dependent inputs.
![accuracy_and_loss_curve](https://github.com/user-attachments/assets/42813d73-6f17-415f-87ab-01ad98d07742)
![audio_wave_and_spectogram_form](https://github.com/user-attachments/assets/4cc8b0f0-3430-4a3e-9878-6bc7c88fbb31)
![classification_report](https://github.com/user-attachments/assets/6de276ab-53e7-44de-9e60-efa8028e88fe)
![confusion_matrix](https://github.com/user-attachments/assets/b079beba-8468-490f-b1af-9b63003b1e0d)
![output screen shot - 2](https://github.com/user-attachments/assets/1b8b7265-4033-4d1c-a1c7-bfd8af6998c9)
![output screen shot - 3](https://github.com/user-attachments/assets/0a0e6bb0-bfad-49ce-9a0f-4c86a4febda8)
![output screen shot - 4](https://github.com/user-attachments/assets/10e9f2cd-c075-475d-a4c5-66ffe450f99f)
![Output_screenshot-2](![Output_screenshot-3](https://github.com/user-attachments/assets/47bba34f-6d6f-4a46-acb5-9fcee59b38d8)
https://github.com/user-attachments/assets/13696cca-5492-4159-a70e-4c51bb39c001)
