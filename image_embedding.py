import os
import sys
import tensorflow as tf
import tensorflow.logging as log
import vgg16

from utils import load_image

tf.flags.DEFINE_string("image",  None, "image filename")
tf.flags.DEFINE_string("images_path", None,
                       "root path containing images to embed")
tf.flags.DEFINE_string("output", "embeddings.csv", "Output embedding file")
FLAGS = tf.flags.FLAGS


def net():
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.Vgg16(images)
    return images, net.embedding


def write_embedding(emb, f, label=None):
    if label:
        f.write(label + ",")
    f.write(",".join([str(x) for x in emb]))
    f.write("\n")


def batch_embed(img_path, output):
    input, embedding = net()
    n = 0
    with open(output, "w") as f:
        with tf.Session() as sess:
            for root, _, files in os.walk(img_path):
                for f in files:
                    if "jpg" in f or "jpeg" in f:
                        p = os.path.join(root, f)
                        n += 1
                        try:
                            img = load_image(p)
                            emb = sess.run(embedding, feed_dict={input: img})
                            write_embedding(emb, f, img)
                            log.info("%ld: Wrote %s", n, p)
                        except Exception:
                            log.warn("Failed on %s", p)
    return n


def embed_image(image):
    input, embedding = net()
    with tf.Session() as sess:
        emb = sess.run(embedding, feed_dict={input: image})
    return emb


def main(_):
    if FLAGS.images_path:
        log.info("Starting scanning path %s", FLAGS.images_path)
        n = batch_embed(FLAGS.images_path, FLAGS.output)
        log.info("Done %ld", n)
    elif FLAGS.image:
        image = load_image(FLAGS.image)
        emb = embed_image(image)
        with open(FLAGS.output) as f:
            write_embedding(emb, f)
    else:
        print "Usage: image_embedding.py --image|--path --output"
        sys.exit(1)


if __name__ == "__main__":
    tf.app.run()
