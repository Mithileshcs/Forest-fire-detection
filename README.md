# Forest-fire-detection

Dataset : used dataset from kaggle

# Steps:

1.Import Necessary Libraries: The code begins by importing essential libraries, including TensorFlow for deep learning, scikit-learn for evaluation metrics, Gradio for creating a user interface, and more.

2.Load and Preprocess the Dataset: In this section (not shown in the provided code), you would load your forest fire dataset, which typically contains images of both fire and non-fire scenes. Preprocessing steps include resizing images, normalizing pixel values, and separating them into training and testing sets.

3.Split the Dataset: The dataset is split into training and testing sets to train the model on one portion and evaluate its performance on another.

4.Data Augmentation for Training Set: Data augmentation techniques are applied to the training set. These techniques create variations of the training images by applying transformations such as rotation, flipping, and scaling. This helps the model generalize better and reduce overfitting.

5.Load MobileNetV2 as a Base Model: MobileNetV2, a pre-trained CNN model, is used as a base model. It has already learned useful features from a large dataset, which can be fine-tuned for the specific task of forest fire detection.

6.Build the Custom Head of the Model: A custom head of the model is added on top of MobileNetV2. This head includes layers for global average pooling, fully connected layers, and an output layer. It tailors the model for the binary classification task of detecting forest fires.

7.Compile the Model: The model is compiled with an optimizer (Adam), a loss function (binary cross-entropy), and metrics (accuracy). This defines how the model will be trained.

8.Train the Model: The model is trained on the training data using the fit method. The code iterates through the training data for a specified number of epochs, updating the model's weights to minimize the loss.

9.Evaluate the Model on the Testing Data: After training, the model's performance is evaluated on the testing data to assess how well it generalizes to new, unseen images.

10.Display Classification Report and Confusion Matrix: Classification report and confusion matrix are generated to evaluate the model's performance in detail, providing metrics like precision, recall, and F1-score.

11.Define a Function for Prediction Using Gradio: A function for making predictions using the trained model is defined. This function takes an image as input and returns the model's prediction.

12.Create a Gradio Interface for Making Predictions: A Gradio interface is created to allow users to upload an image and receive predictions from the model in real-time.

13.Launch the Gradio Interface: The Gradio interface is launched, making the forest fire detection model accessible to users through a user-friendly web interface.

This code combines deep learning, data preprocessing, model training, evaluation, and user interface creation to build a forest fire detection system using a CNN model. Users can upload images, and the model provides predictions on whether the image contains a forest fire or not.



# Output

<img width="1239" alt="output-1" src="https://github.com/Mithileshcs/Forest-fire-detection/assets/94213698/7c60ae20-2694-400a-aefb-60be364b4244">





