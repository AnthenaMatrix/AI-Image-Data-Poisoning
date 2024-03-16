import numpy as np
import cv2

def add_noise(image, epsilon):
    # Generate random noise with the same shape as the image
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=image.shape)

    # Add noise to the image
    noisy_image = image + noise

    # Clip the pixel values to the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def save_poisoned_image(image_path, poisoned_image):
    # Save the poisoned image to a file
    cv2.imwrite("poisoned_" + image_path, poisoned_image)
    print("Poisoned image saved successfully.")

# Example usage:
if __name__ == "__main__":
    # Define the image path
    image_path = "original_image.jpg"

    # Load the original image
    original_image = cv2.imread(image_path)

    # Set the epsilon value for the noise (adjust as needed)
    epsilon = 10

    # Add adversarial noise to the original image
    poisoned_image = add_noise(original_image, epsilon)

    # Save the poisoned image
    save_poisoned_image(image_path, poisoned_image)
