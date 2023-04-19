# First Section: Importing Libraries
import os
import requests
from bs4 import BeautifulSoup
import cv2


# Second Section: Declare important variables
google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"

user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}

# Third Section: Build the main function
saved_folder = 'images_test'


def main():
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    download_images()


# Fourth Section: Build the download function
def download_images():
    data = input('What are you looking for? ')
    n_images = int(input('How many images do you want? '))

    print('searching...')

    search_url = google_image + 'q=' + data

    response = requests.get(search_url, headers=user_agent)

    html = response.text

    soup = BeautifulSoup(html, 'html.parser')

    results = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

    count = 1
    links = []
    for result in results:
        try:
            link = result['data-src']
            links.append(link)
            count += 1
            if(count > n_images):
                break

        except KeyError:
            continue

    print(f"Downloading {len(links)} images...")

    for i, link in enumerate(links):
        response = requests.get(link)

        image_name = saved_folder + '/' + data + str(i+1) + '.jpg'

        with open(image_name, 'wb') as fh:
            fh.write(response.content)
    print('Done!!')

# Fifth Section: Run your code
if __name__ == "__main__":
    main()
    
    


# #################################################################
# # RESIZE IMAGES
# # define the input and output directories
# input_dir = 'D:\Documents\B.Tech\B.Tech_Third_Year\Sem02\Machine_Learning\Assignment05\images_train'
# output_dir = 'D:\Documents\B.Tech\B.Tech_Third_Year\Sem02\Machine_Learning\Assignment05\images\Kangaroo'

# # define the target size
# target_size = (224, 224)

# # loop through all images in the input directory
# for filename in os.listdir(input_dir):
#     # read the image using OpenCV
#     img = cv2.imread(os.path.join(input_dir, filename))
    
#     # resize the image
#     img_resized = cv2.resize(img, target_size)
    
#     # save the resized image to the output directory
#     cv2.imwrite(os.path.join(output_dir, filename), img_resized)
