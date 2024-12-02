import tensorflow as tf
import tensorflow_datasets as tfds
import jax


def crop_resize(image, resolution):
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
      image,
      size=(resolution, resolution),
      antialias=True,
      method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)



batch_size = 32
image_size = 128


def get_oxford_flowers(batch_size, image_size):
    dataset_builder = tfds.builder('oxford_flowers102')
    dataset_builder.download_and_prepare()

    def preprocess_fn(d):
        img = d['image']
        img = crop_resize(img, image_size)
        img = tf.image.flip_left_right(img)
        img= tf.image.convert_image_dtype(img, tf.float32)
        return({'image':img})

    # create split for current process 
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'train[{start}:{start + split_size}]'

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds.with_options(options)

    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size , seed=0)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # (local_devices * device_batch_size), height, width, c
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(10)
    return ds
