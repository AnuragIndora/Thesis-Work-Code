import os
import random
import shutil

# Define paths
dataset_folder = 'Dataset/ChildData/OUTPUTFOLDER/'
train_folder = 'Dataset/ChildData/OUTPUTFOLDER/Train/'  
test_folder = 'Dataset/ChildData/OUTPUTFOLDER/Test/'    

# Create the train and test directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# List of emotion categories (subfolders)
categories = ['Fear', 'Sad', 'Angry', 'Neutral', 'Surprised', 'Happiness', 'Disgust']

# Iterate through each emotion category
for category in categories:
    
    category_folder = os.path.join(dataset_folder, category)

    # Ensure the main category folder exists
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
        print(f"Created folder: {category_folder}")
    else:
        print(f"Folder already exists: {category_folder}")
    
    # Create subdirectories for train and test within each category
    category_train_folder = os.path.join(train_folder, category)
    category_test_folder = os.path.join(test_folder, category)
    os.makedirs(category_train_folder, exist_ok=True)
    os.makedirs(category_test_folder, exist_ok=True)

    # List all the images in the category folder
    images = [img for img in os.listdir(category_folder) if img.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Shuffle the list of images to randomize the selection
    random.shuffle(images)

    # Define the ratio for the split (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(images))  # 80% for training
    train_images = images[:train_size]
    test_images = images[train_size:]

    # Move or copy the images to the respective train and test folders
    for img in train_images:
        src = os.path.join(category_folder, img)
        dst = os.path.join(category_train_folder, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(category_folder, img)
        dst = os.path.join(category_test_folder, img)
        shutil.copy(src, dst)

    print(f"{category} - Train images: {len(train_images)}")
    print(f"{category} - Test images: {len(test_images)}")

print("Dataset split completed!")
