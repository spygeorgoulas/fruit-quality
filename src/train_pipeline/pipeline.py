import argparse


if __name__ == "__main__":
    # Define the argument parser to read in the input and output folder paths
    parser = argparse.ArgumentParser(description='Preprocess and augment fruit images')
    parser.add_argument('-i', '--input_folder_path', type=str, required=True, help='Path to the dataset folder')

    # Parse the arguments
    args = parser.parse_args()