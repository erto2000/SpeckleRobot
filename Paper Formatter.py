from PIL import Image, ImageDraw, ImageFont

# Open the four images
image1 = Image.open("/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/makale_resimler/Resnet50_norm_train.png")
image2 = Image.open("/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/makale_resimler/Resnet18_norm_train.png")
image3 = Image.open("/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/makale_resimler/Mobile_norm_train.png")
image4 = Image.open("/Users/mahirdemir/Desktop/pyhon_vs/git_interact/SpeckleRobot/makale_resimler/Squeeze_norm_train.png")

# Resize images if needed to ensure they have the same dimensions
# You can skip this step if all images have the same dimensions
# For example:
# image1 = image1.resize((width, height))

# Create a new blank image with dimensions for the grid
width, height = image1.size
combined_image = Image.new("RGB", (2 * width, 2 * height))

# Paste each image into the grid
combined_image.paste(image1, (0, 0))
combined_image.paste(image2, (width, 0))
combined_image.paste(image3, (0, height))
combined_image.paste(image4, (width, height))

# Save the combined image
combined_image.save("combined_image.png")

# Display the combined image
combined_image.show()
