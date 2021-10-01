import tensorflow as tf
import numpy as np
import skimage.io
import itertools
import os
import bz2
import argparse
import scipy
import skimage.transform
import time
import matplotlib.pyplot as plt

plt.switch_backend('agg')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

CONTENT_LAYERS = ['4_1','5_1']
LOCAL_STYLE_LAYERS = ['3_1','4_1']
GLOBAL_STYLE_LAYERS=['2_1','3_1','4_1','5_1']


def conv2d(input_tensor, kernel, bias):
    kernel = np.transpose(kernel, [2, 3, 1, 0])
    x = tf.pad(input_tensor, [[0,0], [1,1], [1,1], [0,0]])
    x = tf.nn.conv2d(x, tf.constant(kernel), (1,1,1,1), 'VALID')
    x = tf.nn.bias_add(x, tf.constant(bias))
    return tf.nn.relu(x)

def avg_pooling(input_tensor, size=2):
    return tf.nn.pool(input_tensor, [size, size], 'AVG', 'VALID', strides=[size, size])

def norm(arr):
    n, *shape = arr.shape
    lst = []
    for i in range(n):
        v = arr[i, :].flatten()
        v /= np.sqrt(sum(v**2))
        lst.append(np.reshape(v, shape))
    return lst

def build_base_net(input_tensor):
    vgg19_file = os.path.join(os.path.dirname(__file__), 'vgg19.pkl.bz2')
    assert os.path.exists(vgg19_file), ("Model file with pre-trained convolution layers not found. Download here: "
        +"https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2")

    data = np.load(bz2.open(vgg19_file, 'rb'))
    k = 0
    net = {}
    # network divided into two parts，main and map，main downsamples the image，map dowsamples the semantic map
    net['img'] = input_tensor
    net['conv1_1'] = conv2d(net['img'], data[k], data[k+1])
    k += 2
    net['conv1_2'] = conv2d(net['conv1_1'], data[k], data[k+1])
    k += 2
    # average pooling without padding
    net['pool1']   = avg_pooling(net['conv1_2'])
    net['conv2_1'] = conv2d(net['pool1'], data[k], data[k+1])
    k += 2
    net['conv2_2'] = conv2d(net['conv2_1'], data[k], data[k+1])
    k += 2
    net['pool2']   = avg_pooling(net['conv2_2'])
    net['conv3_1'] = conv2d(net['pool2'], data[k], data[k+1])
    k += 2
    net['conv3_2'] = conv2d(net['conv3_1'], data[k], data[k+1])
    k += 2
    net['conv3_3'] = conv2d(net['conv3_2'], data[k], data[k+1])
    k += 2
    net['conv3_4'] = conv2d(net['conv3_3'], data[k], data[k+1])
    k += 2
    net['pool3']   = avg_pooling(net['conv3_4'])
    net['conv4_1'] = conv2d(net['pool3'], data[k], data[k+1])
    k += 2
    net['conv4_2'] = conv2d(net['conv4_1'], data[k], data[k+1])
    k += 2
    net['conv4_3'] = conv2d(net['conv4_2'], data[k], data[k+1])
    k += 2
    net['conv4_4'] = conv2d(net['conv4_3'], data[k], data[k+1])
    k += 2
    net['pool4']   = avg_pooling(net['conv4_4'])
    net['conv5_1'] = conv2d(net['pool4'], data[k], data[k+1])
    k += 2
    net['conv5_2'] = conv2d(net['conv5_1'], data[k], data[k+1])
    k += 2
    net['conv5_3'] = conv2d(net['conv5_2'], data[k], data[k+1])
    k += 2
    net['conv5_4'] = conv2d(net['conv5_3'], data[k], data[k+1])
    k += 2
    net['main'] = net['conv5_4']

    return net


def extract_target_data(content, style):
    pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))
    # local style patches extracting
    input_tensor = style-pixel_mean
    net = build_base_net(input_tensor)
    local_features = [net['conv'+layer] for layer in LOCAL_STYLE_LAYERS]
    
    tensors = []
    for f in local_features:
        dim = f.get_shape()[-1].value
        # x = (batch, height, width, patches)
        x = tf.extract_image_patches(f, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
        # x = (-1, patch_heigth, patch_width, channles)
        tensors.append(tf.reshape(x, (-1, 3, 3, dim)))
       
    # content features
    input_tensor = content-pixel_mean
    net = build_base_net(input_tensor) 
    content_features = [net['conv'+layer] for layer in CONTENT_LAYERS]
    content_data = []
    
    # feature correlations
    input_tensor = style-pixel_mean
    net = build_base_net(input_tensor)    
    global_features = [net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
    global_gram = []

    for f in global_features:
        N=int(f.shape[3])
        M=int(f.shape[1]*f.shape[2])
        f=tf.reshape(f,(M,N))   
        global_gram.append(tf.matmul(tf.transpose(f),f))
    global_data = []
    
    patches = []
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for t in tensors:
            patches.append(t.eval())
        for c in content_features:
            content_data.append(c.eval())
        for g in global_gram:
            global_data.append(g.eval())

    return content_data,patches,global_data

        
def format_and_norm(arr):
    norm = arr/np.sqrt(np.sum(arr**2))
    return norm


class Model(object):
    def __init__(self, args, content, style, stylized, hist_sim):
        self.args = args
        if len(args.device)>3 and args.device[:3]=='gpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device[3:]
        elif args.device=='cpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))

        self.content = np.expand_dims(content, 0).astype(np.float32)
        self.style = np.expand_dims(style, 0).astype(np.float32)
        self.stylized= np.expand_dims(stylized, 0).astype(np.float32)
        
        # get target content features, local patches, global feature correlations
        self.content_data, self.local_data, self.global_data= extract_target_data(self.content, self.style)
        tf.reset_default_graph()
        
        self.net = build_base_net(self.stylized-self.pixel_mean)

        self.content_features = [self.net['conv'+layer] for layer in CONTENT_LAYERS]
        self.local_features = [self.net['conv'+layer] for layer in LOCAL_STYLE_LAYERS]
        self.global_features = [self.net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
        
        # local pattern similarity
        self.local_sim1 = 0
        self.local_sim2 = 0
        for i in range(len(LOCAL_STYLE_LAYERS)):
            sem = self.local_features[i]
            patches = tf.extract_image_patches(sem, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
            patches = tf.reshape(patches, (-1, 3, 3, sem.shape[-1].value))        
        
            p1 = tf.sqrt(tf.reduce_sum(patches**2,[1,2,3]))
            p1 = tf.reshape(p1, [-1,1,1,1])
            norm_patch = patches/p1
            norm_patch = tf.reshape(norm_patch, [patches.shape[0].value,-1])
           
            p2 = tf.sqrt(tf.reduce_sum(self.local_data[i]**2,[1,2,3]))
            p2 = tf.reshape(p2, [-1,1,1,1])
            norm_target = self.local_data[i]/p2
            norm_target = tf.reshape(norm_target, [self.local_data[i].shape[0], -1])

            sim = tf.matmul(norm_patch, tf.transpose(norm_target))
            max_ind = tf.argmax(sim, axis=-1)
            max_ind = tf.reshape(max_ind, [-1])
            target_patches = tf.gather(self.local_data[i], max_ind)
            
            # compute the number of different style patches in style image
            s_sim = tf.matmul(norm_target, tf.transpose(norm_target))
            s_max_ind = tf.argmax(s_sim, axis=-1)
            s_max_ind = tf.reshape(s_max_ind, [-1])
            
            
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                category_x = len(set(max_ind.eval()))
                category_s = len(set(s_max_ind.eval()))


            p3 = tf.sqrt(tf.reduce_sum(target_patches**2,[1,2,3]))
            p3 = tf.reshape(p3, [-1,1,1,1])
            target_norm_patch = target_patches/p3
            target_norm_patch = tf.reshape(target_norm_patch, [target_patches.shape[0].value, -1])
            
            all_sim = tf.matmul(norm_patch, tf.transpose(target_norm_patch)) 

            self.local_sim1 += tf.reduce_mean(tf.diag_part(all_sim))
            #self.local_sim2 += len(sett)/max_ind.shape[0].value
            self.local_sim2 += category_x/category_s

        weight_of_part1 = 0.5
        self.local_sim = weight_of_part1*(self.local_sim1/len(LOCAL_STYLE_LAYERS)) + (1-weight_of_part1)*(self.local_sim2/len(LOCAL_STYLE_LAYERS))
        
        
        # content fidelity similarity
        self.content_sim = 0
        for i in range(len(CONTENT_LAYERS)):
            sem = self.content_features[i]
            sem_target = self.content_data[i]
            stylized_content_norm = sem/tf.sqrt(tf.reduce_sum(sem**2))
            target_content_norm = sem_target/tf.sqrt(tf.reduce_sum(sem_target**2))
            stylized_content = tf.reshape(stylized_content_norm, [-1,sem.shape[1]*sem.shape[2]*sem.shape[3]])
            target_content = tf.reshape(target_content_norm, [sem_target.shape[1]*sem_target.shape[2]*sem_target.shape[3], -1])
            self.content_sim += tf.reduce_mean(tf.matmul(stylized_content, target_content))

        self.content_sim = self.content_sim/len(CONTENT_LAYERS)


        
        # global effect similarity
        self.global_gram = []
        for f in self.global_features:
            N=int(f.shape[3])
            M=int(f.shape[1]*f.shape[2])
            f=tf.reshape(f,(M,N))   
            self.global_gram.append(tf.matmul(tf.transpose(f),f))

        self.global_sim = 0
        for i in range(len(GLOBAL_STYLE_LAYERS)):
            sem = self.global_gram[i]
            sem_target = self.global_data[i]
            stylized_global_norm = sem/tf.sqrt(tf.reduce_sum(sem**2))
            target_global_norm = sem_target/tf.sqrt(tf.reduce_sum(sem_target**2))
            stylized_global = tf.reshape(stylized_global_norm, [-1,sem.shape[0]*sem.shape[1]])
            target_global = tf.reshape(target_global_norm, [sem_target.shape[0]*sem_target.shape[1], -1])
            self.global_sim += tf.reduce_mean(tf.matmul(stylized_global, target_global))

        weight_of_gram = 0.5
        self.global_sim = weight_of_gram*self.global_sim/len(GLOBAL_STYLE_LAYERS) + (1-weight_of_gram)*hist_sim

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            print('content fidelity:%f, global effect:%f, local patterns:%f.'%
                            (self.content_sim.eval(), self.global_sim.eval(), self.local_sim.eval()))

    

def main():
    parser = argparse.ArgumentParser(description='evaluate the quality of neural style transfer.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument

    add_arg('--content',        default=None, type=str,    help='Content image path.')
    add_arg('--style',          default=None, type=str,    help='Style image path.')
    add_arg('--stylized',          default=None, type=str,    help='Stylized image path.')
    add_arg('--device',         default='cpu', type=str,    help='devices: "gpu"(default: all gpu) or "gpui"(e.g. gpu0) or "cpu" ')
    
    
    args = parser.parse_args()
    
    content = skimage.io.imread(args.content)
    style = skimage.io.imread(args.style)
    stylized = skimage.io.imread(args.stylized)
    
    if stylized.shape[0] != content.shape[0] or stylized.shape[1] != content.shape[1]:
        stylized = skimage.transform.resize(stylized,(content.shape[0],content.shape[1]))
        style = skimage.transform.resize(style,(content.shape[0],content.shape[1]))

    # color histogram similarity
    hist_sim = 0
    for i in range(3):
        n_style,_,_ = plt.hist(style[:,:,i].flatten(), bins=128)
        n_stylized,_,_ = plt.hist(stylized[:,:,i].flatten(), bins=128)
        #norm = max(max(n_style),max(n_stylized))
        #hist_sim += 1 - np.mean(abs(n_style-n_stylized)/norm)
       
        n_style = n_style/np.sqrt(np.sum(n_style**2))
        n_stylized = n_stylized/np.sqrt(np.sum(n_stylized**2))
        n_style = np.reshape(n_style, [1,-1])
        n_stylized = np.reshape(n_stylized, [-1,1])
        hist_sim += np.mean(np.dot(n_style, n_stylized))
    hist_sim = hist_sim / 3
   # print (hist_sim)

    model = Model(args, content, style, stylized, hist_sim)


if __name__ == '__main__':
    tic = time.time()
    main()
    print ("all time:%.4f"%(time.time()-tic))
