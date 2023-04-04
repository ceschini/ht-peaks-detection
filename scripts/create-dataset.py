import random
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
import cv2 as cv
import pandas as pd


def random_coords(width, height):
    return int(random.random() * width), int(random.random() * height)


def draw_random_lines(numLines, image, dims):
    for _ in range(numLines):
        x0, y0 = random_coords(dims[0], dims[1])
        x1, y1 = random_coords(dims[0], dims[1])
        image[draw.line(x0, y0, x1, y1)] = 255

    return image

# super noisy sinusoids approach 2
# changing randomly sinusoid parameters
# for each line determined by numNoise to avoid overlap
# adding rotation so the lines can take
# more positions other than being horizontal


def rotate_points(x, y, angle_degrees, origin):
    angle_radians = np.deg2rad(angle_degrees)
    ox, oy = origin
    px, py = x - ox, y - oy
    qx = ox + px * np.cos(angle_radians) - py * np.sin(angle_radians)
    qy = oy + px * np.sin(angle_radians) + py * np.cos(angle_radians)
    return qx, qy


def sinusoid_noise2(numNoise, image, dims, num_points=100,
                    amplitude_range=(5, 15), frequency_range=(1, 4),
                    noise_factor=5, angle_range=(0, 360), line_thickness=0.5):

    height, width = dims

    for _ in range(numNoise):
        amplitude = np.random.randint(*amplitude_range)
        frequency = np.random.randint(*frequency_range)
        vertical_shift = np.random.randint(0, height)
        angle = np.random.randint(*angle_range)

        x = np.linspace(0, width, num_points)
        y = amplitude * np.sin(frequency * 2 * np.pi *
                               x / width) + vertical_shift
        noisy_x = x + \
            np.random.randint(-noise_factor, noise_factor, size=num_points)
        noisy_y = y + \
            np.random.randint(-noise_factor, noise_factor, size=num_points)

        origin = (width // 2, height // 2)
        noisy_x, noisy_y = rotate_points(noisy_x, noisy_y, angle, origin)

        noisy_x = np.clip(noisy_x, 0, width - 1).astype(int)
        noisy_y = np.clip(noisy_y, 0, height - 1).astype(int)

        for i in range(len(noisy_x) - 1):
            r0, c0 = noisy_y[i], noisy_x[i]
            r1, c1 = noisy_y[i + 1], noisy_x[i + 1]

            rr, cc = draw.line(r0, c0, r1, c1)
            for r, c in zip(rr, cc):
                rr_disk, cc_disk = draw.disk(
                    (r, c), line_thickness, shape=image.shape)
                image[rr_disk, cc_disk] = 255

    return image


def generate_image(numLines=1, numNoise=1, dims=(200, 200)):
    blank = np.zeros(dims)
    img = draw_random_lines(numLines, blank, dims)
    img = sinusoid_noise2(numNoise, img, dims)
    return img.astype('uint8')


def get_hough_transform(image):
    tested_angles = np.linspace(0, np.pi, 360, endpoint=False)
    hspace, angles, dists = hough_line(image, theta=tested_angles)
    _, theta, rho = hough_line_peaks(hspace, angles, dists)
    coords = []
    for rho, theta in zip(rho, theta):
        coords.append((rho, theta))
    ht = np.log(1 + hspace)
    return ht, coords


def get_pixel_coords(lines):
    coords = []
    for rho, theta in lines:
        t = theta * 360 / np.pi
        r = rho + 283
        coords.append((t, r))
    return coords


def draw_detected_lines(image, coords):
    result = np.zeros(image.shape)
    # calculate the x,y coordinates of the line and draw it on the image
    for rho, theta in coords:
        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)
        x1 = int(x0 + 1000 * (-np.sin(theta)))
        y1 = int(y0 + 1000 * (np.cos(theta)))
        x2 = int(x0 - 1000 * (-np.sin(theta)))
        y2 = int(y0 - 1000 * (np.cos(theta)))
        cv.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)
    plt.imshow(result, cmap=cm.gray)
    plt.show()


def create_dataset_image(numLines, numNoise, dims, fname):
    img = generate_image(numLines, numNoise, dims)
    print('saving generated image...')
    plt.imsave(fname=f"./data/{fname}.png", arr=img, cmap='gray', format='png')
    print('image saved!')
    ht, lines = get_hough_transform(img)
    print('saving hough transform...')
    plt.imsave(fname=f"./data/{fname}-ht.png",
               arr=ht, cmap='gray', format='png')
    print('hough transform saved!')
    labels = get_pixel_coords(lines)
    return lines, labels


# def add_to_dataframe(fname, lines, labels, df=None):
#     new_line = pd.DataFrame([[fname, lines, labels]], columns=[
#                             'name', 'polars', 'labels'])
#     new_df = pd.concat([df, new_line])
#     return new_df

def add_to_dataframe(image_id, labels, df=None):
    for label in labels:
        x = label[0]
        y = label[1]
        new_line = pd.DataFrame([[image_id, x, y]], columns=['image_id', 'x', 'y'])
        new_df = pd.concat([df, new_line])

    return new_df


def make_annotation(fname, folder, labels):
    # Creating XML structure
    annon = ET.Element('annotation')
    folder_el = ET.SubElement(annon, 'folder')
    filename_el = ET.SubElement(annon, 'filename')
    path_el = ET.SubElement(annon, 'path')
    source_el = ET.SubElement(annon, 'source')
    source_database_el = ET.SubElement(source_el, 'database')
    size_el = ET.SubElement(annon, 'size')
    size_width_el = ET.SubElement(size_el, 'width')
    size_height_el = ET.SubElement(size_el, 'height')
    size_depth_el = ET.SubElement(size_el, 'depth')
    segmented_el = ET.SubElement(annon, 'segmented')

    # Populating XML elements
    folder_el.text = folder
    filename_el.text = fname
    path_el.text = f'{folder}/{fname}.png'
    source_database_el.text = 'Unknown'
    size_width_el.text = '360'
    size_height_el.text = '567'
    size_depth_el.text = '3'
    segmented_el.text = '0'

    for i, label in enumerate(labels):
        object_el = ET.SubElement(annon, 'object')
        object_name_el = ET.SubElement(object_el, 'name')
        object_pose_el = ET.SubElement(object_el, 'pose')
        object_truncated_el = ET.SubElement(object_el, 'truncated')
        object_difficult_el = ET.SubElement(object_el, 'difficult')
        bndbox_el = ET.SubElement(object_el, 'bndbox')
        bndbox_xmin_el = ET.SubElement(bndbox_el, 'xmin')
        bndbox_ymin_el = ET.SubElement(bndbox_el, 'ymin')
        bndbox_xmax_el = ET.SubElement(bndbox_el, 'xmax')
        bndbox_ymax_el = ET.SubElement(bndbox_el, 'ymax')
        object_name_el.text = f'peak_{i + 1}'
        object_pose_el.text = 'Unspecified'
        object_truncated_el.text = '0'
        object_difficult_el.text = '0'
        x_coord = int(label[0])
        y_coord = int(label[1])
        bndbox_xmin_el.text = f'{x_coord - 5}'
        bndbox_ymin_el.text = f'{y_coord - 5}'
        bndbox_xmax_el.text = f'{x_coord + 5}'
        bndbox_ymax_el.text = f'{y_coord + 5}'

    # converting xml to bytes
    b_xml = ET.tostring(annon)

    # flushing data to file stream
    with open(f'./data/{fname}.xml', 'wb') as f:
        f.write(b_xml)

    return 1


def main(numImages, numLines, numNoise, dims):
    print('#### CREATING DATASET ####')
    dims = (dims, dims)
    # csv = pd.DataFrame(columns=['name', 'polars', 'labels'])
    csv = pd.DataFrame(  # new wheat
        columns=['image_id', 'x', 'y']
        )
    for i in range(numImages):
        filename = f'{numLines}l_{numNoise}n-{i+1}'
        folder = f'{numLines}l_{numNoise}n'
        print(f'** generating image {filename} **')
        lines, labels = create_dataset_image(
            numLines, numNoise, dims, filename)
        print('generating pascal VOC annotation...')
        make_annotation(filename, folder, labels)
        print('adding image data to csv...')
        # csv = add_to_dataframe(filename, lines, labels, csv)
        csv = add_to_dataframe(filename, labels, csv)
        print('image data added to csv!\n')
    print('saving final csv...')
    csv.to_csv(f'./data/{numLines}l_{numNoise}n.csv')
    print('final csv saved!')
    print('#### DATASET CREATION DONE ####')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hough Peaks Detector Dataset Generator')
    parser.add_argument('-i', '--images', type=int,
                        required=True, help='number of images')
    parser.add_argument('-l', '--lines', type=int, required=True,
                        help='number of lines on each image')
    parser.add_argument('-n', '--noise', type=int, required=True,
                        help='number of sinusoid noise on each image')
    parser.add_argument('-d', '--dims', type=int, required=True,
                        help='single dimension (pixels) of the square images')
    args = parser.parse_args()
    main(args.images, args.lines, args.noise, args.dims)
