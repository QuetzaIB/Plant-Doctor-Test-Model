import os
import cv2
from skimage.metrics import structural_similarity as ssim
import v8sort
import segement_leaf
import segement_sick
from PIL import Image
import csv
import time
import re


class PicData:
    def __init__(self, image, value1, value2):
        self.image = image
        self.value1 = value1
        self.value2 = value2

# Calculate the aspect ratio and remove individuals whose aspect ratio exceeds the limit directly.
def function_a(image):
    if image is None:
        return 0
    height, width, _ = image.shape
    aspect_ratio = width / height
    if aspect_ratio > 4 or aspect_ratio < 0.25:
        return 0
    else:
        return 1

# Calculate the Laplacian score for the second rating to assess sharpness.
def function_b(image):
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

# Calculate the similarity for the third rating.
def function_c(img1, img2):
    if img1 is None or img2 is None:
        return 0
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(gray1, gray2)
    return ssim_value

# Calculate the total score by combining three ratings.
def score_cultulate(image0, image):
    a = function_a(image)
    b = function_b(image)
    c = function_c(image0, image)
    score = a * b

    return c, score

# Find the best quality image in each subfolder within a folder.
def find_best_image(input_folder = "sort", output_folder = "result"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Iterate through the subfolders in the main folder.
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Iterate through each image file in every subfolder.
            image0 = None
            flag = 1
            pic_data_list = []
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)
                image = cv2.imread(image_path)
                if flag != 0:
                    image0 = image
                    flag = 0
                if image is not None:
                    c = function_c(image0, image)
                    score = function_a(image)*function_b(image)
                    data_instance = PicData(image, c, score)
                    pic_data_list.append(data_instance)

            total_value1 = sum(data_instance.value1 for data_instance in pic_data_list)
            average_value1 = total_value1 / len(pic_data_list)

            custom_data_list = [data_instance for data_instance in pic_data_list if
                                data_instance.value1 >= average_value1]

            max_value2 = float('-inf')
            max_image = None
            max_image_file = ""

            for data_instance in custom_data_list:
                if data_instance.value2 > max_value2:
                    max_value2 = data_instance.value2
                    max_image = data_instance.image
                    max_image_file = image_file

            if max_image is not None:
                max_image_file_name, _ = os.path.splitext(max_image_file)
                result_path = os.path.join(output_folder, f"{subfolder}_best_{max_image_file_name}.png")
                cv2.imwrite(result_path, max_image)
                print(f"Saved the sharpest image from {subfolder} as {result_path}")


def generate_txt_with_percentage(result_dir, percentages, leaf_values, sick_values, output_file):
    file_names = os.listdir(result_dir)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "ID", "Frame", "Percentage", "Leaf Value", "Sick Value"])
        for file_name, percentage, leaf, sick in zip(file_names, percentages, leaf_values, sick_values):
            match = re.match(r"(\d+)_best_\d+_(\d+).png", file_name)
            if match:
                id_value = match.group(1)
                frame_value = match.group(2)
            else:
                id_value = "Unknown"
                frame_value = "Unknown"
            percentage = round(percentage, 4)
            writer.writerow([file_name, id_value, frame_value, percentage, leaf, sick])

# Return the largest number that is divisible by 32.
def make_divisible_by_32(value):
    return (value + 31) // 32 * 32

# Process the image to make its dimensions divisible by 32.
def update_images_to_divisible_by_32(directory):
    for file_name in os.listdir(directory):
        # Construct full file path
        file_path = os.path.join(directory, file_name)

        # Open an image file
        with Image.open(file_path) as img:
            # Get original dimensions
            width, height = img.size

            # Calculate new dimensions
            new_width = make_divisible_by_32(width)
            new_height = make_divisible_by_32(height)

            # Resize image to new dimensions
            resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

            # Save the resized image back to the same path
            resized_img.save(file_path)


if __name__ == "__main__":
    cut_time_s = time.time()
    v8sort.track_cut_video(input_path="demo.mp4")
    cut_time_e = time.time()
    time1 = cut_time_e - cut_time_s
    print(time1)

    find_time_s = time.time()
    find_best_image()
    find_time_e = time.time()
    time2 = find_time_e - find_time_s
    print(time2)

    sick_time_s = time.time()
    update_images_to_divisible_by_32("./result")
    leaf = []
    sick = []
    leaf = segement_leaf.segment_leaf()
    #print(leaf)
    sick = segement_sick.segment_sick()
    #print(sick)
    percentages = []
    for sick_count, leaf_count in zip(sick, leaf):
        if leaf_count != 0:
            percentage = (sick_count / leaf_count) * 100
        else:
            percentage = 0
        percentages.append(percentage)

    print(percentages)
    sick_time_e = time.time()
    time3 = sick_time_e - sick_time_s
    print(time3)

    generate_txt_with_percentage('./result', percentages, leaf, sick, output_file='output.csv')
