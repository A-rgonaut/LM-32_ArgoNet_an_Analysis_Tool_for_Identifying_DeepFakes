
import cv2
import pickle
import argparse
import Morpher as mp
import DeepFakeDetector as dfd

# python main.py <img_path> --dump --test
# python main.py dataset\real\Aaron_Pena\Aaron_Pena_0001.jpg --dump --test

def main():
    parser = argparse.ArgumentParser(description="DeepFakeDetector class")
    parser.add_argument("path", type=str, help="Path of the image to testing")
    parser.add_argument("--dump", action="store_true", help="Show in command line the progress and make a CSV dump")
    parser.add_argument("--test", action="store_true", help="")
    args = parser.parse_args()

    try:
        with open("best_model.pkl", "rb") as f:
            model, param = pickle.load(f)

        detector = dfd.DeepFakeDetector("dataset")
        detector.model = model
        detector.param = param

    except FileNotFoundError:
        try:
            detector = dfd.DeepFakeDetector("dataset")
        except Exception as e:
            print("Error occour: ",e)

    #morpher = mp.Morpher(cv2.imread("src/coppia3_1.jpg"), cv2.imread("src/coppia3_2.jpg"), 0.5)
    #cv2.imwrite("test.jpg", morpher.morph(0.5, False))        
    
    detector.get_model(args.dump, args.test)

    print(detector.test_model(args.path))

if __name__ == "__main__":
    main()