import cv2
import numpy as np
import os
from extract import extract
from masking import mask


def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    # image plate
    plate = extract(image, points)

    # blur plate image
    kernel = np.ones((100, 100), np.float32) / 10000
    blured_plate = cv2.filter2D(plate, -1, kernel)

    # mask photo with image of plate
    output = mask(image, points, blured_plate)
    return output


if __name__ == '__main__':
    test_images = ['test_plates//1a4ea979-day_13562', 'test_plates//1f193783-day_00271',
                   'test_plates//1ef77e57-day_07335']
    for i in range(len(test_images)):
        img = os.path.join(os.getcwd(), test_images[i])

        with open(img + '.txt') as file:
            for line in file:
                coordinates = line.rstrip().split(' ')

        coordinates.pop(0)
        in_image = cv2.imread(img + '.jpg')
        cover_image = cv2.imread('kntu.jpg')

        result = blur(in_image, np.array(coordinates, dtype=np.float32))
        cv2.imshow('blured plate image', result)
        cv2.waitKey()
