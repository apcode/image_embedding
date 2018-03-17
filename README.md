# Image Embedding

Generate embeddings for images using Vgg16.

Uses the first fully connected layer after the conv layers. So
embeddings are 4096 vectors per image.

## Usage

Can either generate for a single image:

    python image_embedding.py --image=image.jpg --output=embedding.csv

The output embedding is a csv file of 4096 floats.

Or it can walk a directory tree converting all jpg images it finds
into one embedding file.

    python image_embedding.py --images_path=./path --output=embeddings.csv

The output format is one line per image embedding. Each 4096 vector is
prepended with the image filename.
