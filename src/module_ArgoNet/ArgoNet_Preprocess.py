
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor

class Preprocess():
    """
    A class for preprocessing image datasets, specifically designed for the ArgoNet project.
    This class handles image processing tasks including resizing, splitting into train/validation/test sets,
    and organizing the processed images into a structured directory hierarchy.
        dataset_path (str): Path to the root directory containing the original dataset
        output_path (str): Path where the processed dataset will be saved
        target_size (Tuple[int, int]): Desired dimensions (height, width) for the processed images
        splits (Tuple[int, int, int]): Ratios for train, validation, and test splits respectively
    Attributes:
        dataset_path (str): Storage for the input dataset path
        output_path (str): Storage for the processed data output path
        target_size (Tuple[int, int]): Target dimensions for image resizing
        splits (dict): Dictionary containing split ratios for train, validation, and test sets
    """

    def __init__(self, dataset_path: str, output_path: str, target_size: Tuple[int, int], splits: Tuple[int, int, int]):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.target_size = target_size 
        self.splits = {'train': splits[0], 'val': splits[1], 'test': splits[2]}

    def __setup_gpus(self):
        """
        Configure GPU settings for TensorFlow.
        Enables memory growth for available GPUs to prevent memory allocation issues.
        """
    
        gpus = tf.config.list_physical_devices('GPU')
    
        if gpus:
            # Abilita crescita memoria GPU
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

    def __build_dir_struct(self):
        """
        Creates the directory structure for organizing real and fake images.
        The method creates directories for each split (e.g. train, validation, test) 
        and within each split creates subdirectories for real and fake images.
        The directory structure will be:
        output_path/
            ├── split_1/
            │   ├── real/
            │   └── fake/
            ├── split_2/
            │   ├── real/
            │   └── fake/
            └── ...
        Directory creation is safe and won't raise errors if directories already exist.
        """
        
        for split in self.splits.keys():
            for label in ['real', 'fake']:
                os.makedirs(os.path.join(self.output_path, split, label), exist_ok=True)

    def __get_img_paths(self, label: str) -> List[str]:
        """
        Gets all image paths for a given label from the dataset directory.
        Args:
            label (str): The label/class name corresponding to a subdirectory in the dataset path.
        Returns:
            List[str]: A list of full paths to all image files with .png, .jpg, or .jpeg extensions 
            in the label's directory and its subdirectories.
        Example:
            If dataset_path is '/data' and label is 'cat', this will return paths like:
            ['/data/cat/img1.jpg', '/data/cat/folder/img2.png', ...]
        """
        
        label_path = os.path.join(self.dataset_path, label)
        img_paths = []

        for root, _, files in os.walk(label_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(root, file))

        return img_paths
    
    def __splitting(self, img_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Split the provided image paths into training, validation, and test sets according to predefined split ratios.
        Args:
            img_paths : List[str]
                List of paths to image files that need to be split
        Returns:
            Tuple[List[str],List[str],List[str]]:
            - train_paths: List of image paths for training set
            - val_paths: List of image paths for validation set 
            - test_paths: List of image paths for test set
        """    

        # Calculate split sizes
        n_total = len(img_paths)
        n_train = int(n_total * self.splits['train'])
        n_val = int(n_total * self.splits['val'])
        
        # Split paths
        train_paths = img_paths[:n_train]
        val_paths = img_paths[n_train:n_train + n_val]
        test_paths = img_paths[n_train + n_val:]
        
        return train_paths, val_paths, test_paths

    def __processing(self, src_path: str, dst_path: str) -> None:
        """
        Process and resize a single image.
        This method reads an image from the source path, resizes it to the target size
        specified in the class initialization, and saves the processed image to the 
        destination path.
        Args:
            src_path (str): Source path of the input image to be processed
            dst_path (str): Destination path where the processed image will be saved
        Raises:
            Exception: If there are errors during image processing, reading or writing
        """
        

        try:
            # Read image
            img = tf.io.read_file(src_path)
            img = tf.image.decode_image(img, channels=3)
            
            # Resize image
            img = tf.image.resize(img, self.target_size)
            img = tf.cast(img, tf.uint8)
            
            # Save processed image
            tf.io.write_file(dst_path, tf.image.encode_jpeg(img))

        except Exception as e:
            print(f"Error processing {src_path}: {str(e)}")

    def __downsampling(self, max_samples: int = 50000) -> None:
        """
        Downsamples and processes the image dataset by splitting it into train, validation, and test sets.
        This method handles both 'real' and 'fake' image categories, applying random sampling if the number
        of images exceeds the maximum sample size. Images are processed and saved to the corresponding
        output directories using parallel processing.
        Args:
            max_samples (int, optional): Maximum number of samples to process per category. Defaults to 50000.
        Notes:
            - Images are randomly shuffled before processing
            - If number of images exceeds max_samples, random sampling without replacement is performed
            - Processing is done in parallel using ThreadPoolExecutor
            - Output images are saved as JPG files in their respective category and split folders
        """
        

        for label in ['real', 'fake']:
            print(f"\nProcessing {label} images...")
            
            # Get and shuffle image paths
            img_paths = self.__get_img_paths(label)
            np.random.shuffle(img_paths)

            if len(img_paths) > max_samples:
                img_paths = np.random.choice(img_paths, max_samples, replace=False)
            
            train_paths, val_paths, test_paths = self.__splitting(img_paths)
            
            # Process each split
            splits_data = {'train': train_paths, 'val': val_paths, 'test': test_paths}
            
            for split_name, paths in splits_data.items():
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = []

                    for path in paths:

                        filename = os.path.basename(path)
                        base_filename = os.path.splitext(filename)[0] + '.jpg'

                        dest_path = os.path.join(self.output_path, 
                                                 split_name, 
                                                 label,
                                                 base_filename
                                                 )
                        
                        futures.append(executor.submit(self.__processing, path, dest_path))
                    
                    # Monitor progress
                    for _ in tqdm(futures, desc=f"Processing {split_name} set"): _.result()

    def start(self):
        """
        Initializes and starts the preprocessing pipeline for the ArgoNet dataset.
        This method performs the following steps:
        1. Configures GPU settings and memory allocations
        2. Creates necessary directory structure for processed data
        3. Executes downsampling of images
        The method should be called after instantiating the preprocessing class to begin data preparation.
        """
        

        # Setup GPU and memory configurations
        self.__setup_gpus()
        self.__setup_memory()
        # Build directory structure
        self.__build_dir_struct()
        # Downsample and process images
        self.__downsampling()

if __name__ == "__main__":
    
    dataset_path = "dataset"
    output_path = "dataset_processed"
    
    preprocessor = Preprocess(dataset_path, output_path, (224, 224), (0.7, 0.15, 0.15))
    preprocessor.start()