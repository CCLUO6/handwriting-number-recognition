import os
import numpy as np
import cv2



if __name__ == "__main__":
    # Example usage
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, one_hot_label=False)
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)