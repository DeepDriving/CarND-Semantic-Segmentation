import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image,keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    
    #Encoder
    #Fully connected 1x1 convoluation with layer 7 output
    #layer7_fcn = tf.layers.conv2d(vgg_layer7_out, layer7_filters, (1,1), (1,1), padding='SAME',
     #                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    layer7_fcn = tf.layers.conv2d(vgg_layer7_out, num_classes, (1,1), (1,1), 
                                kernel_initializer=tf.truncated_normal_initializer( stddev=0.001 ))
    #Decoder

    #1st Upsampling
    layer7_upscale = tf.layers.conv2d_transpose(layer7_fcn, num_classes,(4,4), (2,2), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.001) )

    #Fully Connected  1x1 convoluation with layer 4 output
    layer4_fcn = tf.layers.conv2d(vgg_layer4_out, num_classes, (1,1), (1,1),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.001) )

    #Skip connection
    layer4_skip = tf.add(layer7_upscale, layer4_fcn)

    #2nd upsampling
    layer4_upscale = tf.layers.conv2d_transpose(layer4_skip, num_classes, (4,4), (2,2), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))

    #Full conneced 1x1 convolution with layer 3 output
    layer3_fcn = tf.layers.conv2d(vgg_layer3_out, num_classes, (1,1), (1,1), 
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))

    #Skip connection
    layer3_skip = tf.add(layer4_upscale, layer3_fcn)

    #Output by upsampling last layer
    output = tf.layers.conv2d_transpose(layer3_skip, num_classes, (16,16), (8,8), padding='SAME',
                                kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))


    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function


    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels)
    loss = tf.reduce_mean(entropy)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):

        for image, gt_image in get_batches_fn(batch_size):
            feed_dict = {
                input_image: image,
                correct_label: gt_image,
                keep_prob: 0.5,
                learning_rate: 0.0001                     
            }
        _,loss = sess.run([train_op, cross_entropy_loss], feed_dict= feed_dict)
        print("Epoch: {}/{}, Loss {}".format(epoch, epochs, loss))   
tests.test_train_nn(train_nn)

def save_model(sess):
    saver = tf.train.Saver()
    saver.save(sess,'/media/annie.guan/DATA/Data/Kitti/KittiSeg/DATA/model.ckpt')
    saver.export_meta_graph('/media/annie.guan/DATA/Data/Kitti/KittiSeg/DATA/model.meta')
    tf.train.write_graph(sess.graph_def, "/media/annie.guan/DATA/Data/Kitti/KittiSeg/DATA/", "model.pb", False)

def run():
    num_classes = 2
    epochs = 80
    batch_size = 8
    image_shape = (160, 576)
    data_dir = '/media/annie.guan/DATA/Data/Kitti/KittiSeg/DATA'
    runs_dir = '/media/annie.guan/DATA/Data/Kitti/KittiSeg/RUNS'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

        
        
        input_image,keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,vgg_path)

        output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        label = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 2])
        learning_rate = tf.placeholder(dtype=tf.float32)
        
        logits, train_op, loss = optimize(output,label,learning_rate,num_classes)

        sess.run(tf.global_variables_initializer())
        
        train_nn(sess,epochs,batch_size, get_batches_fn, train_op, loss,input_image, 
                 label, keep_prob, learning_rate)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        #save the model
        save_model(sess)



if __name__ == '__main__':
    run()
