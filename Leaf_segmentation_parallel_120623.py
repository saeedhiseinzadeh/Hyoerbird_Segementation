import sys
import os
import cv2 as cv
import spectral as spec
import numpy as np
from plantcv import plantcv as pcv
#from plantcv.parallel import WorkflowInputs
#import re
import multiprocessing
import csv
spec.settings.envi_support_nonlowercase_params = True



def extract_wavelengths (hdr_file_path):
    wavelengths = []
    with open(hdr_file_path, 'r') as file:
        # Flag to mark when we are reading wavelengths
        reading_wavelengths = False
        for line in file:
            # Check if this line marks the start of wavelengths
            if line.strip().lower().startswith('wavelength ='):
                reading_wavelengths = True
                continue
            
            # Check for the end of the wavelengths list
            if reading_wavelengths and '}' in line:
                break

            # Read and store wavelengths
            if reading_wavelengths:
                # Remove commas, split, and convert to float
                values = line.replace(',', '').strip().split()
                wavelengths.extend([float(value) for value in values])

    return wavelengths

# Function to compute average of a slice excluding zeros
def average_slice_excluding_zeros(slice_data):
    non_zero_values = slice_data[slice_data != 0]
    if non_zero_values.size == 0:
        return 0  # Return 0 if all values are zero
    return np.mean(non_zero_values)

def process_file(file_name, white_ref, black_ref, threshold, disk_images):
    try:
        full_file_path = os.path.join(os.getcwd(), file_name)  # Construct the full file path

        if file_name.endswith('.raw') and full_file_path not in [white_ref, black_ref]:
            hdr_file_path = full_file_path.replace('.raw', '.hdr')
            png_filename = file_name.replace('.raw', '.png')

            if os.path.exists(hdr_file_path) and not os.path.exists(os.path.join(disk_images, png_filename)):
                try:
                    # Read the files in envi format 
                    raw_file = pcv.readimage(full_file_path, mode='envi')

                    # Calibrate the data with the white and dark reference
                    # Calibrate the data with the white and dark reference
                    Calibrated_data = pcv.hyperspectral.calibrate(raw_data=raw_file, white_reference=white_ref, dark_reference=dark_ref)
                    Cal_data = Calibrated_data.array_data
                    #define the threshold
                    Threshold = threshold
                    binary_mask = cv.threshold(Cal_data, Threshold, 1, cv.THRESH_BINARY)[1]

                    # morphological functions to make mask into a solid mask.
                    # note: a morphological approach could be taken where the edge of the mask
                    # is more organically shaped and not smooth like the cirlce mask.
                    # Additionally, the morphological functions could likely be condensed/
                    #made more efficient as I am only using the binary + morphed mask to find
                    # the centroid of the leaf disk so I can draw a circle.
                    kernel = np.ones((3,3), np.uint8)
                    kernel2 = np.ones((9,9),np.uint8)
                    kernel3 = np.ones((5,5), np.uint8)
                    kernel4 = np.ones((11,11), np.uint8)
                    # note: I chose band 158 to use for masking. The leaf disk appeared to contrast
                    #       well with the background at this band.
                    m1 = cv.morphologyEx(binary_mask[:,:,158], cv.MORPH_CLOSE, kernel2)
                    m2 = cv.erode(m1, kernel, iterations = 2)
                    m3 = cv.dilate(m2, kernel2, iterations = 4)
                    m4 = cv.morphologyEx(m3, cv.MORPH_CLOSE, kernel4)

                    # connected component labeling to calculate centroid
                    num_labels, labeled_mask, stats, centroids = cv.connectedComponentsWithStats(m4.astype(np.uint8))

                    # Find the label of the largest connected component (excluding the background label 0)
                    largest_label = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

                    # Get the centroid coordinates of the largest connected component
                    centroid_x, centroid_y = centroids[largest_label]

                    # Create a blank image for drawing the circle
                    height, width = binary_mask.shape[:2]
                    circle_image1 = np.zeros((height, width), dtype=np.uint8)
                    circle_image2 = np.zeros((height, width), dtype=np.uint8) #For outline purposes, do not change.
                    # Draw a circle around the centroid of the largest connected component
                    radius1 = 890 # Adjust the radius of the circle as needed
                    radius2 = 930 #For outline purposes, do not change.
                    cv.circle(circle_image1, (int(centroid_x), int(centroid_y)), radius1, 1, thickness=-1) #Draws on blank image
                    #cv.circle(circle_image2, (int(centroid_x), int(centroid_y)), radius2, 255, thickness=-1) #For outline purposes, do not change.

                    # Replace .raw extension with .hdr to get the corresponding hdr file
                    img = spec.open_image(hdr_file_path)

                    masked_circle = Cal_data[:,:,158]*(circle_image1/1)
                    masked_img = img[:,:,:]*(circle_image1/255)[:,:,np.newaxis]

                    # Visualization and saving of the image
                    # Full path for the image
                    save_path = os.path.join("disk_images", (str(file_name[:-4]) + ".png"))
                    # Save the image
                    cv.imwrite(save_path, masked_circle * 255)

                    #How to mask a 3D array
                    # Assuming data_3d is your 3D array (shape: (2209, 2196, 233))
                    # and mask is your 2D mask (shape: (2209, 2196))
                    # Reshape the mask to make it (2209, 2196, 1)
                    mask_reshaped = circle_image1[:, :, np.newaxis]
                    # Now apply the mask using broadcasting
                    data = Cal_data * mask_reshaped
                    # masked_data is now the 3D array with the mask applied to each slice
                    # Assuming 'data' is your 3D array with shape (2230, 2190, 233)
                    # and 'filename' is the name of the file corresponding to this data
                    # Compute the average of each slice, excluding zeros

                    averages = [average_slice_excluding_zeros(data[:, :, i]) for i in range(data.shape[2])]
                    # Prepare the row for the CSV file
                    row = [file_name[:-4]] + averages
                    #the directory of the hdr file
                    hdr_path= os.path.join('.', hdr_file_path)
                    # Check if CSV file exists and if not, create it with headers
                    csv_file_path = os.path.join(os.getcwd(), os.path.basename(os.getcwd()) + '.csv')
                    if not os.path.exists(csv_file_path):
                        with open(csv_file_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            headers = ['Filename'] + extract_wavelengths (hdr_path)
                            writer.writerow(headers)

                    # Append the data to the CSV file
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                    # Implement the processing logic here
                    # If there's an error, raise an exception
                    print(f"Successfully processed {file_name}")
                except Exception as process_error:
                    print(f"Error processing {file_name}: {process_error}. Skipping file.")
            else:
                print(f"Skipping {file_name}: Processed before or HDR file is not existed")

    except Exception as e:
        print(f"Error in processing loop for {file_name}: {e}")


def main(white_ref, black_ref, threshold):
    disk_images = 'disk_images'
    white_ref = pcv.readimage(filename=white_ref, mode='envi')
    dark_ref = pcv.readimage(filename=black_ref, mode='envi')

    # Check if the disk_images directory exists, and create it if it doesn't
    if not os.path.exists(disk_images):
        os.makedirs(disk_images)
        print(f"Created directory: {disk_images}")

    # List of all .raw files that are not the white or black reference images
    all_files = [f for f in os.listdir('./') if f.endswith('.raw') and f not in {os.path.basename(white_ref), os.path.basename(black_ref)}]

    # Set up multiprocessing
    num_processes = multiprocessing.cpu_count()  # or set manually to 20

    # Creating a pool of processes
    with multiprocessing.Pool(num_processes) as pool:
        # Start the process pool and pass the necessary arguments
        pool.starmap(process_file, [(f, white_ref, dark_ref, threshold, disk_images) for f in all_files])

    # Optional: Additional code to handle any post-processing or cleanup after all files are processed
    # This is a placeholder for any post-processing or cleanup code you might have.
    # For example, you can log the completion of processing, check for any errors, or
    # perform aggregations or summaries of the results.
    print("All files have been processed.")

def main(white_ref, black_ref, threshold):
    disk_images = 'disk_images'
    white_ref = pcv.readimage(filename=white_ref, mode='envi')
    dark_ref = pcv.readimage(filename=black_ref, mode='envi')

    # ... [rest of your setup code]

    all_files = [f for f in os.listdir('./') if f.endswith('.raw') and os.path.join(os.getcwd(), f) not in [white_ref, black_ref]]

    # Determine the number of processes
    num_processes = multiprocessing.cpu_count()  # or set manually to 20

    # Create a pool of processes
    with multiprocessing.Pool(num_processes) as pool:
        # Using starmap to pass multiple arguments to the function
        pool.starmap(process_file, [(file_name, white_ref, black_ref, threshold, disk_images) for file_name in all_files])

if __name__ == "__main__":
    # Default values
    default_threshold = 0.2
    # Extract arguments
    white_ref = sys.argv[1]
    black_ref = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else default_threshold
    
    # Validate the band value
    if threshold < 0 or threshold > 1:
        print(f"Warning: Band value {threshold} is out of range. It should be between 0 and 1.")
        sys.exit(1)

    main(white_ref, black_ref, threshold)
