
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

np.random.seed(42)

class DeepFakeDetector():
    """
    DeepFakeDetector is a class designed to detect deepfake images by leveraging machine learning models
    and image processing techniques.

    Available subpackages
    -
     - os 
     - cv2 
     - pickle  
     - numpy 
     - pandas
     - skimage
     - matplotlib
     - tqdm
     - sklearn

    Attributes
    -
    model (object)
        The trained machine learning model used for deepfake detection.
    param (dict)
        A dictionary containing the parameters of the best model, including the model type,
        method used for feature extraction, standardization status, scaler object, and accuracy.
    paths (dict)
        A dictionary containing the paths to the fake and real image datasets.
    faceClassifier (cv2.CascadeClassifier)
        A pre-trained face classifier used for detecting faces in images.

    Methods
    -
    - get_model()
    - __crop_face(img: cv2.typing.MatLike)
    - __local_binary_pattern(img: cv2.typing.MatLike, method: str)
    - __make_dataset(method: str)
    - __load_dataset(method: str)
    - __split_dataset(method: str)
    - __train_models(dump: bool = False, test: bool = False)
    - test_model(img_path: str)
    """

    def __init__(self, path  : str = './dataset'):
        self.model = None
        self.param = {"model": None, "method": None, "std": None, "scaler": None, "accuracy": 0.0}
        self.paths = {"fake": path + '/fake', "real": path + '/real'}
        try:
            first_folder = next(os.walk(self.paths["fake"]))[1][0]
            first_img = os.listdir(os.path.join(self.paths["fake"], first_folder))[0]
            self.rows, self.cols = cv2.imread(os.path.join(self.paths["fake"], first_folder, first_img)).shape[:2]
        except IndexError:
            raise FileNotFoundError("No folders or images found in the fake dataset path.")
        self.faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     
    def get_model(self, dump : bool = False, test : bool = False):
        """
        Retrieves the best trained model and its parameters. If the model does not exist,
        it triggers the training process to create the best model.
        Returns:
            tuple: A tuple containing the best trained model and its parameters.
        """
        if self.model is None:
            print("Model not exists. Training the best model...")
            self.__train_models(dump,test)

        return self.model, self.param
    
    def __crop_face(self, img : cv2.typing.MatLike):
        """
        Crops a single face from the given image using a pre-trained face cascade detector.
        Args:
            img (cv2.typing.MatLike): The input image in which to detect and crop a face.
        Returns:
            cv2.typing.MatLike: The cropped and resized face image if exactly one face is detected.
            None: If no face or more than one face is detected in the image.
        Notes:
            - The face detection is performed using the `detectMultiScale` method of the 
              pre-trained face cascade (`self.faceClassifier`).
            - The detected face is resized to the dimensions specified by `self.cols` and `self.rows`.
        """
        # Detecting the faces in the image
        faces = self.faceClassifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # cropping the face if only one face has been detected
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cropped_face = img[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (self.cols, self.rows))

            return cropped_face

    def __local_binary_pattern(self, img : cv2.typing.MatLike, method : str):
        """
        Computes the Local Binary Pattern (LBP) of an image using the specified method.
        Parameters:
            img (cv2.typing.MatLike): The input image in BGR format.
            method (str): The LBP method to use. Options are:
                - "default": Uses 256 points and a radius of 1.
                - "uniform": Uses 8 points and a radius of 1.
        Returns:
            cv2.typing.MatLike: The computed LBP image.
        Raises:
            ValueError: If an invalid method is provided.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculating the Local Binary Pattern with the required method
        if method == "default":
            return ski.feature.local_binary_pattern(gray_img, 256, 1, method=method)
        elif method == "uniform":
            return ski.feature.local_binary_pattern(gray_img, 8, 1, method=method)
        else:
            raise ValueError("Invalid method. Choose 'default' or 'uniform'.")

    def __make_dataset(self, method : str):
        """
        This method create a dataset for training and testing the DeepFake detector.
        This method processes images from specified paths, extracts features using 
        Local Binary Pattern (LBP), and saves the resulting dataset as a pickle file.
        Args:
            method (str): The LBP method to use for feature extraction.
        Returns:
            np.ndarray: A NumPy array containing the processed dataset. Each row 
            represents an image, with its LBP histogram features and a label 
            (0 for real, 1 for fake).
        Workflow:
            1. Iterates through the paths for real and fake images.
            2. Loads images from the specified directories.
            3. Crops the face region for real images.
            4. Computes the LBP histogram for each image.
            5. Appends the histogram data, label and identity to the dataset.
            6. Saves the dataset to a pickle file named "dataset_<method>.pkl".
        Notes:
            - Only images with extensions ".png" and ".jpg" are processed.
            - If an image cannot be read or processed, it is skipped.
        """
        # Initializing the dataset and the identities counter
        dataset = []
        identity = 0

        # Browsing the real and fake directories
        for label, path in self.paths.items():
            # Browsing the identity directories
            for folder in tqdm(os.listdir(path), desc=f"Loading {label.capitalize()} Images"):
                # Browsing the images of each subject
                for file in os.listdir(os.path.join(path, folder)):
                    # Getting png or jpg images
                    if file.endswith((".png", ".jpg")):
                        image_path = os.path.join(path, folder, file)
                    image = cv2.imread(image_path)

                    # Cropping the face of real labelled images
                    if label == "real":
                        image = self.__crop_face(image)

                    # Processing the image appending to the end of each histogram
                    # his label and the corrispondent identity id
                    if image is not None:
                        lbp = self.__local_binary_pattern(image, method)
                        histogram = np.histogram(lbp.ravel(), bins=np.arange(256))[0]
                        histogram = np.append(histogram, 1 if label == "fake" else 0)
                        histogram = np.append(histogram, identity)
                        dataset.append(histogram)
                identity += 1

        # Saving the dataset
        dataset = np.array(dataset)

        with open("dataset_" + method + ".pkl", "wb") as f:
            pickle.dump(dataset, f)

        return dataset

    def __load_dataset(self, method : str):
        """
        Loads a dataset from a pickle file based on the specified method. If the file
        does not exist, it attempts to create the dataset using the `__make_dataset` method.
        Args:
            method (str): The method name used to identify the dataset file.
        Returns:
            object: The loaded dataset object if successful, or None if an error occurs.
        """
        # Loading the pre-processed dataset of the required method
        try:
            with open("dataset_" + method + ".pkl", "rb") as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            # Pre-processing the dataset if doesn't already exist
            print("Dataset not found. Making dataset...")
            try:
                dataset = self.__make_dataset(method)
            except Exception as e:
                print(f"Error creating dataset: {e}")
                return None
        
        return dataset
    
    def __split_dataset(self, method : str):
        """
        Splits the dataset into training, validation, and test sets based on subjects.
        Args:
            method (str): The method used to load the dataset. This is passed to the 
                          `__load_dataset` function to retrieve the dataset.
        Returns:
            tuple: A tuple containing the following:
                - X_training (numpy.ndarray): Features for the training set.
                - X_validation (numpy.ndarray): Features for the validation set.
                - X_test (numpy.ndarray): Features for the test set.
                - y_training (numpy.ndarray): Labels for the training set.
                - y_validation (numpy.ndarray): Labels for the validation set.
                - y_test (numpy.ndarray): Labels for the test set.
        Notes:
            - The dataset is assumed to have features in all columns except the last two.
            - The second-to-last column is treated as the label, and the last column is 
              treated as the identity.
            - The dataset is split into training, validation, and test sets using 
              `train_test_split` with a fixed random state for reproducibility.
        """
        # Loading pre-processed datasets of their respective methods 
        dataset = self.__load_dataset(method)

        # Making a temporary dataset 2xN of identities and labels
        identities = np.unique(dataset[:, -2:], axis=0)
        
        # Setting the identities as feature and the label vectors
        X, y = identities[:,1], identities[:,0]

        # Splitting twice the temporary dataset by the identities making three sets:
        # 60% training set, 20% validation set, 20% test set
        X_validation, X_test, y_validation, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_training, X_validation, y_training, y_validation = train_test_split(X_validation, y_validation, test_size=0.25, random_state=42)

        # Making three masks of indicies of the respective features of identities in the dataset
        train_mask = np.isin(dataset[:, -1], X_training)
        val_mask = np.isin(dataset[:, -1], X_validation)
        test_mask = np.isin(dataset[:, -1], X_test)
        
        # Reassignment of the sets with the features based on the identities
        X_training = dataset[train_mask, :-2]
        X_validation = dataset[val_mask, :-2]
        X_test = dataset[test_mask, :-2]
        
        y_training = dataset[train_mask, -2]
        y_validation = dataset[val_mask, -2]
        y_test = dataset[test_mask, -2]

        return X_training, X_validation, X_test, y_training, y_validation, y_test

    def __train_models(self, dump : bool = False, test : bool = False):
        """
        Trains multiple machine learning models on the dataset using different configurations
        and evaluates their performance. The best-performing model is saved as an attribute
        of the class. Optionally, results can be saved to a CSV file and the model can be tested
        on a test dataset.
        Args:
            dump (bool, optional): If True, saves the training results to a CSV file and prints
                the validation accuracy and best model details. Defaults to False.
            test (bool, optional): If True, evaluates the best model on the test dataset and
                displays the test accuracy and confusion matrix. Defaults to False.
        Side Effects:
            - Updates the `self.model` attribute with the best-performing model.
            - Updates the `self.param` dictionary with details of the best-performing model.
            - Optionally saves training results to a CSV file if `dump` is True.
            - Optionally displays the test accuracy and confusion matrix if `test` is True.
        """
        results = []
        # Initializing models and iper-parameters
        models = [LinearSVC, LogisticRegression, RandomForestClassifier]
        standardization = [True, False]
        methods = ['default', 'uniform']

        # Making a set of all combinations of the models
        params = [(model, std, method) for model in models for std in standardization for method in methods]
        
        # Initializing pre-processed datasets of their respective methods 
        datasets = {method : self.__split_dataset(method) for method in methods}

        # Iterate the models
        for model, std, method in tqdm(params) if dump else params:
            # Setting random_state, if the models is a LogisticRegression set also 
            # max_iter as 1000 to avoid the standard maximum iteratation error
            if model == LogisticRegression:
                clf = model(random_state=42, max_iter=1000)
            else:
                clf = model(random_state=42)

            # Getting the training and validation sets of their respective method
            X_training, X_validation, _, y_training, y_validation, _ = datasets[method]

            scaler = None
            # Fitting and transform by the scaler if the standardization is True
            if std:
                scaler = StandardScaler().fit(X_training)
                X_training = scaler.transform(X_training)
                X_validation = scaler.transform(X_validation)
            
            # Fitting and predict the sets by the actual model  
            clf.fit(X_training, y_training)
            y_prediction = clf.predict(X_validation)

            # Getting the accuracy of actual model by the validation set
            acc = accuracy_score(y_validation, y_prediction)

            # Setting the new best model if the accuracy of the actual model 
            # is better than the actual best model
            if acc > self.param["accuracy"]:
                self.model = clf
                self.param["model"] = clf.__str__()
                self.param["method"] = method
                self.param["std"] = std
                self.param["scaler"] = scaler
                self.param["accuracy"] = acc

            results.append({
                'Classifier': clf.__str__().split('(')[0],
                'Standardization': 'YES' if std else 'NO',
                'Method': method,
                'Accuracy': acc
            })

        # Saving the best model
        with open("best_model.pkl", "wb") as f:
            pickle.dump((self.model, self.param), f)
        
        # Printing in console the results of all trained models if required
        if dump:
            results_df = pd.DataFrame(results)
            results_df.to_csv("res/results.csv")
            print("\nValidation Accuracy:\n", results_df)
            print("\nBest Model:\n", str(self.param)) 

        # Testing the best model if required
        if test:
            # Getting the test set of his respective method
            _, _, X_test, _, _, y_test = datasets[self.param["method"]]

            # Transform the test set by the scaler if the best model has it
            if self.param["std"]:
                X_test = self.param["scaler"].transform(X_test)

            # Predicting the test set
            y_prediction = self.model.predict(X_test)

            # Getting the accuracy of actual best model by the test set
            acc = accuracy_score(y_test, y_prediction)
            print(f"\nTest Accuracy:\n{acc}")

            # Getting the confusion matrix of actual best model
            cm = confusion_matrix(y_test, y_prediction)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{self.param['model']}\n{self.param['method']} - {'STD' if self.param['std'] else 'NO STD'}", weight='bold')
            plt.show()
      
    def test_model(self, img_path : str):
        """
        Tests the trained model on a given image to determine if it is "Fake" or "Real".
        Args:
            img_path (str): The file path to the image to be tested.
        Returns:
            str: "Fake" if the model predicts the image as fake, otherwise "Real".
        Raises:
            ValueError: If the image is not found or the provided path is invalid.
        Notes:
            - If the model is not already trained, it will automatically train the best model.
            - The image is preprocessed by cropping the face and optionally standardizing it.
            - Local Binary Pattern (LBP) is used to extract features from the image before prediction.
        """
        if self.model is None:
            print("Model not exists. Training the best model...")
            self.__train_models()

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or invalid path.")
        
        img = self.__crop_face(img)
        if img is None:
            raise ValueError("Invalid image. Multiple face detected.")

        lbp = self.__local_binary_pattern(img, self.param["method"])
        data = np.histogram(lbp.ravel(), bins=np.arange(256))[0]

        data = data.reshape(1,-1)
        
        if self.param["std"]:
            data = self.param["scaler"].transform(data)

        prediction = self.model.predict(data)
        return "Fake" if prediction == 1 else "Real"