
import os
import shutil
import argparse
import kagglehub
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential                                                         # type: ignore
from tensorflow.keras.optimizers import Adam                                                    # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator                             # type: ignore
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, Reshape  # type: ignore 
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D              # type: ignore   
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau        # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
import ArgoNet_F1Score
from ArgoNet_Attention import Attention

class ArgoNet():
    """
    ArgoNet - A Convolutional Neural Network for Image Classification
    This class implements a CNN architecture for binary image classification, specifically
    designed to distinguish between real and AI-generated images. It supports optional
    attention mechanisms and provides comprehensive training, evaluation, and prediction
    capabilities.
    Attributes:
        TARGET_SIZE (tuple): Target image dimensions (224,224)
        CLASS_MODE (str): Classification mode, set to "binary"
        BATCH_SIZE (int): Batch size for training (64)
        EPOCHS (int): Number of training epochs (20)
        SEED (int): Random seed for reproducibility (42)
        model (tf.keras.Model): The neural network model
        history (tf.keras.callbacks.History): Training history
        attention (bool): Flag for using attention mechanism
    Methods:
        - set_batch_size(batch_size: int): Set the batch size for training.
        - set_epochs(epochs: int): Set the number of training epochs.
        - set_seed(seed: int): Set the random seed for reproducibility.
        - set_attention(attention: bool): Enable or disable attention mechanism.
        - start(path: str = 'ArgoNet.h5', test: bool = True): Initialize and start the model, optionally performing testing evaluation.
        - evaluate_image(img_path: str) -> float: Evaluate a single image and return classification probability.
    Usage Example:
        model = ArgoNet()
        model.set_attention(True)
        model.start()
        probability = model.evaluate_image("path/to/image.jpg")
        - The model can be trained from scratch or loaded from a saved state
        - Supports GPU acceleration when available
        - Includes comprehensive metrics tracking and visualization
        - Implements early stopping and learning rate reduction
        - Uses data augmentation techniques during training
    """

    TARGET_SIZE = (224,224)
    CLASS_MODE = "binary"
    BATCH_SIZE = 64
    EPOCHS = 20
    SEED = 42

    def __init__(self):
        self.model = None
        self.history = None
        self.attention = False

    def __setup_gpus(self):
        """
        Configure GPU settings for TensorFlow.
        Enables memory growth for available GPUs to prevent memory allocation issues.
        """
    
        gpus = tf.config.list_physical_devices('GPU')
    
        if gpus:
            # Enabling memory growth for available GPUs
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    ⚙️  GPU[{i}]: {gpu.name}.")

            print(f"    🟢 GPU/s configurated: {len(gpus)} GPU/s available.")
        else:
            print(f"    🟡 GPU/s not found. Using CPU.")
        
    def __setup_memory(self):
        """
        Clear TensorFlow backend session to free memory.
        """
        
        print(f"    🧹 Cleaning memory...")
        tf.keras.backend.clear_session()
        print(f"    🟢 Memory configurated.")
        
    def __setup_callbacks(self):
        """
        Configure training callbacks for the model.
        Returns:
            list: List of callbacks including EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau
        """
        
        # Early Stopping setup
        early_stopping = EarlyStopping(
            monitor = 'val_loss',
            patience = 3,
            restore_best_weights = True,
            verbose = 1
        )
        
        # Model Checkpoint setup
        model_checkpoint = ModelCheckpoint(
            'ArgoNet.h5',
            monitor = 'val_loss',
            save_best_only = True,
            save_weights_only = False,
            verbose = 1
        )
        
        # Reduce Learning Rate seup
        reduce_lr = ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.2,
            patience = 1,
            min_lr = 1e-7,
            verbose = 1
        )

        print(f"    🟢 Callbacks configurated.")
        
        return [early_stopping, model_checkpoint, reduce_lr]

    def __split_dataset(self, path : str, split_ratio : list[int, int, int], aug : bool = False): 
        """
        Split and prepare the dataset for training, validation, and testing.
        This method handles dataset preparation including downloading if not found locally, 
        and creates data generators for train, validation and test sets with optional augmentation.

        Parameters
        -
        path : str
            Path to the dataset directory containing train, val and test subdirectories
        split_ratio : list[int, int, int]
            List of three numbers representing train, validation and test split ratios that must sum to 1.0
        aug : bool, optional
            Whether to apply data augmentation to training set (default is False)

        Returns
        -
        tuple
            A tuple containing three ImageDataGenerator objects:
            - train_gen: Generator for training data (with optional augmentation)
            - val_gen: Generator for validation data 
            - test_gen: Generator for test data
        
        Raises
        -
        ValueError
            If split_ratio values don't sum to 1.0
        
        Notes
        
        The data augmentation, when enabled, includes:
        - Rotation (±20 degrees)
        - Width/height shifts (±20%)
        - Shear transformation (±20%)
        - Zoom (±20%)
        - Horizontal flips
        All images are rescaled by 1/255 regardless of augmentation setting.
        """
        
        # Check if the split_ratio is valid
        if sum(split_ratio) != 1.0:
            raise ValueError("Splits must sum to 1.0 for for train, val, and test.")
        
        # Check if the dataset path exists
        if os.path.exists(path):
            print("🟢 Dataset found.")         
        else:
            print("🟡 Dataset not found. Downloading dataset...")
            src_path = kagglehub.dataset_download("argonautex/ff-genai-splitted")
            dst_path = os.getcwd()

            # Move the dataset to the current working directory
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dst_path, item)
                shutil.move(s,d)
            shutil.rmtree(src_path)

        # Create data generators for train, validation and test sets
        data_gen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                fill_mode = 'nearest'
            ) if aug else ImageDataGenerator(rescale = 1./255)

        train_gen = data_gen.flow_from_directory(
            path + '/train',
            target_size = self.TARGET_SIZE,
            batch_size = self.BATCH_SIZE,
            class_mode = self.CLASS_MODE,
            shuffle = True,
            seed = self.SEED
        )

        data_gen = ImageDataGenerator(rescale = 1./255)

        val_gen = data_gen.flow_from_directory(
            path + '/val',
            target_size = self.TARGET_SIZE,
            batch_size = self.BATCH_SIZE,
            class_mode = self.CLASS_MODE,
            shuffle = False,
            seed = self.SEED
        )

        test_gen = data_gen.flow_from_directory(
            path + '/test',
            target_size = self.TARGET_SIZE,
            batch_size = self.BATCH_SIZE,
            class_mode = self.CLASS_MODE,
            shuffle = False,
            seed = self.SEED
        )
                
        return train_gen, val_gen, test_gen

    def __build_model(self):
        """Builds and returns a Sequential CNN model with optional attention mechanism.
        The model is designed for image classification with a binary output. It consists of:
        - Multiple Conv2D layers with dilated convolutions
        - MaxPooling layers for dimensionality reduction
        - Dense layers with dropout for classification
        - Optional attention mechanism
        The model architecture differs based on the attention flag:
        - With attention: Includes an attention layer after the convolutional features
        - Without attention: Uses global average pooling after convolutions
        Returns:
            tf.keras.Sequential: A compiled CNN model with the following structure:
                - Input shape: (224, 224, 3)
                - Convolutional layers with dilated convolutions
                - MaxPooling layers
                - Dense layers with dropout
                - Output: Single neuron with sigmoid activation for binary classification
        Notes:
            The model uses:
            - ReLU activation in hidden layers
            - Sigmoid activation in output layer
            - Dropout rate of 0.5 in dense layers
            - Dilated convolutions with varying rates
        """
        
        if self.attention:
            return Sequential([
                Input(shape=(224, 224, 3)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(32, 3, padding='same', activation='relu', dilation_rate=2),
                MaxPool2D(strides=(2,2)),
                Conv2D(32, 3, padding='same', activation='relu', dilation_rate=2),
                MaxPool2D(strides=(2,2)),
                Conv2D(16, 3, padding='same', activation='relu', dilation_rate=1),
                MaxPool2D(strides=(2,2)),
                Flatten(),
                Dense(128, activation='relu'),
                Reshape((1, 128)),
                Attention(embed_dim=128, num_heads=8, name='attention'),
                GlobalAveragePooling1D(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        else:
            return Sequential([
                Input(shape=(224, 224, 3)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(64, 5, padding='same', activation='relu', dilation_rate=3),
                MaxPool2D(strides=(2,2)),
                Conv2D(32, 3, padding='same', activation='relu', dilation_rate=2),
                MaxPool2D(strides=(2,2)),
                Conv2D(32, 3, padding='same', activation='relu', dilation_rate=2),
                MaxPool2D(strides=(2,2)),
                Conv2D(16, 3, padding='same', activation='relu', dilation_rate=1),
                MaxPool2D(strides=(2,2)),
                GlobalAveragePooling2D(),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

    def __compile_model(self, model):
        """
        Compile the model with optimizer, loss function, and metrics.
        Args:
            model: Keras model to compile
        Returns:
            tf.keras.Model: Compiled model
        """
        
        # Compile the model with Adam optimizer, binary crossentropy loss, and metrics
        model.compile(
            optimizer = Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'), 
                    # Custom F1 Score metric
                    ArgoNet_F1Score.F1Score(name='f1_score')]
        )

        return model
    
    def __evaluate_model(self, model, test_gen):
        """
        Evaluate model performance on test set.
        Args:
            model: Trained Keras model
            test_gen: Test data generator
        Returns:
            dict: Dictionary containing evaluation metrics and confusion matrix
        """
        
        # Prediction and true labels
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        # Accuracy, Precision, Recall, F1 Score and Confusion Matrix
        evaluation = model.evaluate(test_gen)
        class_names = list(test_gen.class_indices.keys())
        cm = self.__plot_confusion_matrix(y_true, y_pred, class_names)

        # Prepare results dictionary
        res = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'precision': evaluation[2],
            'recall': evaluation[3],
            'f1_score': evaluation[4],
            'confusion_matrix': cm}
        
        print(f"    📈 Scores:")
        for key, value in res.items():
            if isinstance(value, (float, int)):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
        
        # Classification report
        print(f"    📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        return res

    def __train_model(self):
        """
        Train the ArgoNet model for binary classification.
        This method handles the complete training pipeline including:
        - GPU and memory setup
        - Dataset splitting and configuration
        - Callback setup
        - Model building and compilation
        - Model training
        The method includes comprehensive error handling and progress monitoring through console outputs.
        Returns:
            tf.keras.Model: The trained model loaded from saved weights, with appropriate custom objects
                           based on whether attention mechanism is used.
        Raises:
            Exception: Various exceptions can be raised during setup phases (GPU, memory, dataset, etc.)
                      Each exception is caught and printed before program termination.
        Notes:
            - The training uses a 70-15-15 split for training, validation and test sets
            - Training runs for 20 epochs
            - The model is saved to 'ArgoNet.h5' during training
            - Progress is displayed with detailed console output
        """
        
        print(f"\n" + "-" * 50 + "\n🎯 Training ArgoNet for binary classification\n" + "-" * 50)
        
        # Setups
        print(f"\n>>> Configuration: GPU/s...")
        try: self.__setup_gpus() 
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)

        print(f"\n>>> Configuration: Memory...")
        try: self.__setup_memory()
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)
        
        print(f"\n>>> Configuration: Dataset...")
        try: training_gen, validation_gen, test_gen = self.__split_dataset("./dataset", [0.7, 0.15, 0.15], aug = False)
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)
        
        print(f"    ⚙️  Training samples: {training_gen.samples}")
        print(f"    ⚙️  Validation samples: {validation_gen.samples}")
        print(f"    ⚙️  Test samples: {test_gen.samples}")
        print(f"    ⚙️  Classes: {training_gen.class_indices}")
        print(f"    🟢 Dataset configurated.")

        print(f"\n>>> Configuration: Callbacks...")
        try: callbacks = self.__setup_callbacks()
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)        
        
        # Create and compile the model
        print(f"\n>>> Configuration: Model...")
        try: 
            model = self.__build_model()
            model = self.__compile_model(model)
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)      
        
        model.summary()
        print(f"    🟢 Model configurated.")
        
        # Training
        print("\n>>> Start training...\n" + "*" * 50 + "\n")
        
        self.history = model.fit(
            training_gen,
            epochs = 20,
            validation_data = validation_gen,
            callbacks = callbacks,
            verbose = 1 
        )

        print("\n"+ "*" * 50)

        print(f"\n" + "-" * 50 + "\n🟢 Model trained successfully!\n" + "-" * 50)

        if self.attention:
            return tf.keras.models.load_model("ArgoNet.h5", custom_objects={'F1Score': ArgoNet_F1Score.F1Score,'Attention': Attention})
        else:
            return tf.keras.models.load_model("ArgoNet.h5", custom_objects={'F1Score': ArgoNet_F1Score.F1Score})
        
    def __load_model(self, path : str):
        """
        Load a pre-trained model from the specified path or train a new one if not found.
        Args:
            path (str): Path to the saved model file.
        Returns:
            tf.keras.Model: The loaded or newly trained model.
        Notes:
            - If a model exists at the specified path, it will be loaded with custom objects
              (F1Score and optionally Attention if self.attention is True)
            - If no model is found at the path, a new model will be trained using __train_model()
            - Prints status messages indicating whether model was loaded or needs training
        """

        # Check if the model file exists
        if os.path.exists(path):
            if self.attention:
                self.model = tf.keras.models.load_model(path, custom_objects={'F1Score': ArgoNet_F1Score.F1Score,'Attention': Attention})
            else:
                self.model = tf.keras.models.load_model(path, custom_objects={'F1Score': ArgoNet_F1Score.F1Score})
            print(f"    🟢 Model loaded.")
            return self.model
        else:
            print(f"    🟡 Model not found. Training a new model...")
            return self.__train_model()

    def __plot_training_history(self, history):
        """
        Plots and saves the training history metrics of the model.
        This method creates a 2x2 grid of plots showing the training and validation metrics over epochs:
        - Loss
        - Accuracy
        - F1 Score 
        - Precision & Recall
        The plots are saved as a high resolution PNG file.
        Args:
            history: A Keras History object containing training history metrics.
                    Expected to have the following metrics:
                    - loss, val_loss
                    - accuracy, val_accuracy 
                    - f1_score, val_f1_score
                    - precision, val_precision
                    - recall, val_recall
        Returns:
            None. The plot is saved as 'training_history.png' in the current directory.
        """
        
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(history.history['f1_score'], label='Training F1')
        axes[1, 0].plot(history.history['val_f1_score'], label='Validation F1')
        axes[1, 0].set_title('Model F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision e Recall
        axes[1, 1].plot(history.history['precision'], label='Training Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def __plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot and save confusion matrix for classification results.
        Parameters:
            y_true (array-like): Ground truth (correct) target values.
            y_pred (array-like): Estimated targets as returned by a classifier.
            class_names (list): List of strings with the names of the classes.
        Returns:
            numpy.ndarray: The confusion matrix. Entry [i,j] is the number 
            of observations known to be in group i and predicted to be in group j.
        Notes:
            - Saves the confusion matrix plot as 'confusion_matrix.png' in the current directory
            - Uses seaborn's heatmap for visualization
            - The plot is automatically closed after saving
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm

    def set_batch_size(self, batch_size : int):
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self.BATCH_SIZE = batch_size

    def set_epochs(self, epochs : int):
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")
        self.EPOCHS = epochs

    def set_seed(self, seed : int):
        if seed < 0:
            raise ValueError("Seed must be a non-negative integer.")
        self.SEED = seed

    def set_attention(self, attention : bool):
        self.attention = attention

    def start(self, path : str = 'ArgoNet.h5', test : bool = True):
        """
        Initializes and starts the ArgoNet model, optionally performing testing evaluation.
        This method loads a pre-trained model from a file and can perform model evaluation
        on test data, generating performance metrics and visualization plots.
        Parameters:
            path (str, optional): Path to the saved model file. Defaults to 'ArgoNet.h5'.
            test (bool, optional): Whether to perform testing evaluation. Defaults to True.
        Raises:
            Exception: If there is an error loading the model from the specified path.
        The test evaluation, if enabled:
        - Loads and preprocesses test data using ImageDataGenerator
        - Generates performance plots (training history)
        - Evaluates model performance on test data
        - Saves model file and visualization plots
        Files generated:
        - ArgoNet.h5: Saved model file
        - training_history.png: Plot of training metrics
        - confusion_matrix.png: Confusion matrix visualization
        Returns:
            None
        """
        
        print(f"\n🌐 ArgoNet 🌐")
        print(f"\n" + "=" * 50 + "\n🚀 Starting model...\n" + "=" * 50)

        print(f"\n>>> 💾 Loading model...")
        # Attempt to load the model from the specified path
        try: self.model = self.__load_model(path)
        except Exception as e: 
            print(f"    🔴 Error: {e}")
            exit(1)  

        print(f"\n" + "=" * 50 + "\n🟢 Model started successfully!\n" + "=" * 50)

        # If test evaluation is enabled, perform evaluation based on the test set
        if test:
            print(f"\n" + "." * 50 + "\n🔍 Test evaluation\n" + "." * 50)

            data_gen = ImageDataGenerator(rescale = 1./255)

            test_gen = data_gen.flow_from_directory(
            './dataset/test',
                target_size = self.TARGET_SIZE,
                batch_size = self.BATCH_SIZE,
                class_mode = self.CLASS_MODE,
                shuffle = False,
                seed = self.SEED
            )
            print(f"\n>>> 📊 Graphics...")
            self.__plot_training_history(self.history) if self.history is not None else None
            self.__evaluate_model(self.model, test_gen)

            print(f"\n>>> 📁 Saving files...")
            print(f"     - ArgoNet.h5")
            print(f"     - training_history.png")
            print(f"     - confusion_matrix.png")

    def evaluate_image(self, img_path: str) -> float:
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        if self.model is None:
            raise RuntimeError("Model not loaded. Call start() first")
            
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.TARGET_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, 0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        print(f"\n>>> Image evaluation: {img_path}")
        print(f"    Probability of being real: {prediction * 100}%")

        return prediction

if __name__ == "__main__":

    # python3 <model_path> <image_path> --attention --test 2> null
    # python3 ArgoNet.py ArgoNet.h5 test/pizzi.jpg --attention 2> null

    parser = argparse.ArgumentParser(description="CNN Argonet Class")
    parser.add_argument("model_path", type=str, help="Path of trained model. If not exist it will be train.")
    parser.add_argument("image_path", type=str, help="Path of your image to evaluate.")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args() 

    os.system("cls") if os.name == "nt" else os.system("clear")

    ArgoNet = ArgoNet()
    ArgoNet.set_attention(args.attention)
    ArgoNet.start(path = args.model_path, test = args.test)
    
    ArgoNet.evaluate_image(args.image_path) if args.image_path != "none" else None