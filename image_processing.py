import numpy as np                  # Matrix image operations
import cv2                          # Computer vision
import glob                         # File path manipulation
import os
from PIL import Image, ImageColor
from matplotlib import pyplot as plt

def create_directory(path):
    """ Creates empty folder if folder doesn't exist.

    Args:
        path (string): Relative or absolute path for creating a folder
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print('Creating folder structure at:', path)



def remove_if_exists(path):
    """ Removes all files on given path for fresh results
    on every script run.

    Args:
        path (string): Input path for file removal.
    """

    for item in glob.glob(path + '*.jpeg'):
        if os.path.exists(path + '*.jpeg'):
            os.remove(path + '*.jpeg')

def find_cell(image):
    """ Identifies cells in range of lower and upper blue RGB color 
        spectrum.
    Args:
        image (numpy.ndarray): Image loaded through cv2 package.
    Returns:
        [numpy.ndarray]: Returns blue cell mask.
    """
    
    # RGB type blue color interval
    lower_blue = np.array([90,90,150])
    upper_blue = np.array([150,150,255])
    
    # Convert image to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask in range of upper and lower blue
    mask = cv2.inRange(image, lower_blue, upper_blue)

    #plt.imshow(mask)
    #plt.show()

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    filtered_mask = cv2.dilate(opening, kernel, iterations=3)

    return filtered_mask


def crop_image(path):
    """ Crops image around masked part. Mask contains a region of interest 
    (A white cell).
    Args:
        path (string): String which contains image path.

    Returns:
        [numpy.ndarray]: Cropped image in a black frame.
    """

    image = cv2.imread(path)
    mask = find_cell(image)

    #plt.imshow(image)
    #plt.show()
    #plt.imshow(mask)
    #plt.show()

    v =  np.sum(mask, axis=1)
    h =  np.sum(mask, axis=0)

    try:
        nonzero_idx = np.argwhere(h)
        x1 = nonzero_idx[0][0]
        x2 = nonzero_idx[-1][0]
        nonzero_idx = np.argwhere(v)
        y1 = nonzero_idx[0][0]
        y2 = nonzero_idx[-1][0]

        return image[y1:y2, x1:x2]
    except:
        pass



def resize(input_path, output_path, width, height):
    """Resize image to wanted dimensions and image to black background.

    Args:
        input_path (string): Image input path
        output_path (string): Image output file
        width (int): Width in pixels
        height (int): Width in pixels
    """

    image_pil = Image.open(input_path)
    
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # Fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    
    background.convert('RGB').save(output_path, 'jpeg')

    #plt.imshow(background)
    #plt.show()




def run(input_path, output_path):
    """ Collects all functions, iterates through a folder structure, 
    crops images from input path and saves them to output file.
    Args:
        input_path (string): Input images for cropping.
        output_path (string): Output path for saving cropped images.
    """

    create_directory(output_path)
    remove_if_exists(output_path)

    print('Processing', input_path)
    
    bad_crops_list = []

    for path in glob.iglob(input_path + '*.jpeg'):

        image = crop_image(path)

        image_original = cv2.imread(path)
        out = output_path + os.path.basename(path)
        
        try:
            cv2.imwrite (out, image)
            #image.save(str(out), 'jpeg')
            #print('Output path: ',out)
            resize(out, out, 100, 100)
        except:
            bad_crops_list.append(out)
            #print('Bad crop at: ', out)
        
    print('Number of bad crops: ', len(bad_crops_list))

# Train dataset
run('dataset/dataset2-master/images/TRAIN/EOSINOPHIL/', 'dataset/dataset2-master/images_processed/TRAIN/EOSINOPHIL/')
run('dataset/dataset2-master/images/TRAIN/LYMPHOCYTE/', 'dataset/dataset2-master/images_processed/TRAIN/LYMPHOCYTE/')
run('dataset/dataset2-master/images/TRAIN/MONOCYTE/', 'dataset/dataset2-master/images_processed/TRAIN/MONOCYTE/')
run('dataset/dataset2-master/images/TRAIN/NEUTROPHIL/', 'dataset/dataset2-master/images_processed/TRAIN/NEUTROPHIL/')

# Test dataset
run('dataset/dataset2-master/images/TEST/EOSINOPHIL/', 'dataset/dataset2-master/images_processed/TEST/EOSINOPHIL/')
run('dataset/dataset2-master/images/TEST/LYMPHOCYTE/', 'dataset/dataset2-master/images_processed/TEST/LYMPHOCYTE/')
run('dataset/dataset2-master/images/TEST/MONOCYTE/', 'dataset/dataset2-master/images_processed/TEST/MONOCYTE/')
run('dataset/dataset2-master/images/TEST/NEUTROPHIL/', 'dataset/dataset2-master/images_processed/TEST/NEUTROPHIL/')

# Test simple dataset
run('dataset/dataset2-master/images/TEST_SIMPLE/EOSINOPHIL/', 'dataset/dataset2-master/images_processed/TEST_SIMPLE/EOSINOPHIL/')
run('dataset/dataset2-master/images/TEST_SIMPLE/LYMPHOCYTE/', 'dataset/dataset2-master/images_processed/TEST_SIMPLE/LYMPHOCYTE/')
run('dataset/dataset2-master/images/TEST_SIMPLE/MONOCYTE/', 'dataset/dataset2-master/images_processed/TEST_SIMPLE/MONOCYTE/')
run('dataset/dataset2-master/images/TEST_SIMPLE/NEUTROPHIL/', 'dataset/dataset2-master/images_processed/TEST_SIMPLE/NEUTROPHIL/')