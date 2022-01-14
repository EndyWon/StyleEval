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
        x = tf.extract_image_patches(f, (1,3,3,1), (1,1,1,1), (1,1,1,1), 'VALID')
        tensors.append(tf.reshape(x, (-1, 3, 3, dim)))
    
    # global feature correlations  
    global_features = [net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
    global_gram = []

    for f in global_features:
        N=int(f.shape[3])
        M=int(f.shape[1]*f.shape[2])
        f=tf.reshape(f,(M,N))   
        global_gram.append(tf.matmul(tf.transpose(f),f))
    global_data = []

    # content features
    input_tensor = content-pixel_mean
    net = build_base_net(input_tensor)    
    content_features = [net['conv'+layer] for layer in CONTENT_LAYERS]
    content_data=[]
    
    patches=[]
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for t in tensors:
            patches.append(t.eval())
        for c in content_features:
            content_data.append(c.eval())
        for g in global_gram:
            global_data.append(g.eval())

    return content_data,patches,global_data
      


"""MONet"""
class Model(object):
    def __init__(self, args, content, style):
        self.args = args
        if len(args.device)>3 and args.device[:3]=='gpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device[3:]
        elif args.device=='cpu':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.pixel_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1,1,1,3))

        self.content = np.expand_dims(content, 0).astype(np.float32)
        self.style = np.expand_dims(style, 0).astype(np.float32)
        
       
        # get target content features, local patches, global feature correlations
        self.content_data, self.local_data, self.global_data = extract_target_data(self.content, self.style)
        tf.reset_default_graph()
        
        input_tensor = tf.Variable(self.content)
            
        self.net = build_base_net(input_tensor)

        self.content_features = [self.net['conv'+layer] for layer in CONTENT_LAYERS]
        self.local_features = [self.net['conv'+layer] for layer in LOCAL_STYLE_LAYERS]
        self.global_features = [self.net['conv'+layer] for layer in GLOBAL_STYLE_LAYERS]
        
        self.local_loss = 0
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
            
            
            sess1 = tf.Session()
            sess1.run(tf.global_variables_initializer())
            category_x = len(set(max_ind.eval(session=sess1)))
            category_s = len(set(s_max_ind.eval(session=sess1)))
            sess1.close()
        
            # local style loss
            self.local_loss += tf.reduce_mean((patches-target_patches)**2) + (category_x-category_s)**2
        self.local_loss /= len(LOCAL_STYLE_LAYERS)

        
        # content loss
        self.content_loss = 0
        for c, t in zip(self.content_features, self.content_data) :
            self.content_loss += tf.reduce_mean((c-t)**2)
        self.content_loss /= len(CONTENT_LAYERS)
        #self.content_loss *= args.content_weight
        #self.content_loss *= 0.5
        
        
        # color histogram loss
        hist_loss = 0
        for i in range(3):
            n_style,_,_ = plt.hist(style[:,:,i].flatten(), bins=128)
            n_content,_,_ = plt.hist(content[:,:,i].flatten(), bins=128)
            hist_loss += tf.reduce_sum((n_style.astype(np.float32)-n_content.astype(np.float32))**2)
        hist_loss = hist_loss / 3

        # global style loss
        self.global_loss = 0
        for i in range(len(GLOBAL_STYLE_LAYERS)):
            f=self.global_features[i]
            N=int(f.shape[3])
            M=int(f.shape[1]*f.shape[2])
            f=tf.reshape(f,(M,N))  
            gram=tf.matmul(tf.transpose(f),f)
            self.global_loss += tf.reduce_sum(((gram-self.global_data[i])**2)/((2*M*N)**2))
        self.global_loss /= len(GLOBAL_STYLE_LAYERS)
        self.global_loss += hist_loss
        
        self.mini = tf.minimum(self.local_loss, self.global_loss)

        def cond(a):
            return a/self.mini > 1

        def body(a):
            return a/10
  
        self.local_loss = tf.while_loop(cond, body, [self.local_loss])
        self.global_loss = tf.while_loop(cond, body, [self.global_loss])
        

        # total loss
        self.loss = self.content_loss + self.global_loss + self.local_loss
        self.grad = tf.gradients(self.loss, self.net['img'])
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

    def evaluate(self):
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        def func(img):
            self.iter += 1
            current_img = img.reshape(self.content.shape).astype(np.float32) - self.pixel_mean

            feed_dict = {self.net['img']:current_img}
            loss = 0
            grads = 0
            local_loss = 0
            content_loss = 0
            global_loss=0
            sess.run(tf.global_variables_initializer())
            loss, grads, local_loss, content_loss, global_loss, summ= sess.run(
                [self.loss, self.grad, self.local_loss, self.content_loss, self.global_loss, self.merged],
                feed_dict=feed_dict)
            if self.iter % 10 == 0:
                out = current_img + self.pixel_mean
                out = np.squeeze(out)
                out = np.clip(out, 0, 255).astype('uint8')
                skimage.io.imsave('outputs/liwand-%d.jpg'%(self.iter), out)

            print('Epoch:%d,loss:%f,local loss:%f,global loss:%f,content loss:%f.'%
                (self.iter, loss, local_loss, global_loss, content_loss))
            if np.isnan(grads).any():
                raise OverflowError("Optimization diverged; try using a different device or parameters.")

            # Return the data in the right format for L-BFGS.
            return loss, np.array(grads).flatten().astype(np.float64)
        return func

    def run(self):
        args = self.args
        Xn = self.content
            
        self.iter = 0
        # Optimization algorithm needs min and max bounds to prevent divergence.
        data_bounds = np.zeros((np.product(Xn.shape), 2), dtype=np.float64)
        data_bounds[:] = (0.0, 255.0)
        print ("MONet: Start")
        try:
            Xn, *_ = scipy.optimize.fmin_l_bfgs_b(
                            self.evaluate(),
                            Xn.flatten(),
                            bounds=data_bounds,
                            factr=0.0, pgtol=0.0,            # Disable automatic termination, set low threshold.
                            m=5,                             # Maximum correlations kept in memory by algorithm.
                            maxfun=args.iterations,        # Limit number of calls to evaluate().
                            iprint=-1)                       # Handle our own logging of information.
        except OverflowError:
            print("The optimization diverged and NaNs were encountered.",
                    "  - Try using a different `--device` or change the parameters.",
                    "  - Make sure libraries are updated to work around platform bugs.")
        except KeyboardInterrupt:
            print("User canceled.")
        except Exception as e:
            print(e)
            
        print ("MONet: Completed!")

        #self.summary_writer.close()
    

def main():
    parser = argparse.ArgumentParser(description='MONet: transfer style of a image onto a content image.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument

    add_arg('--content',        default=None, type=str,    help='Content image path.')
    add_arg('--style',          default=None, type=str,    help='Style image path.')
    add_arg('--iterations',     default=500, type=int,       help='Number of iterations.')
    add_arg('--device',         default='gpu', type=str,    help='devices: "gpu"(default: all gpu) or "gpui"(e.g. gpu0) or "cpu" ')
    
    args = parser.parse_args()
    
    style = skimage.io.imread(args.style)
    content = skimage.io.imread(args.content)
    
    model = Model(args, content, style)
    model.run()


if __name__ == '__main__':
    tic = time.time()
    main()
    print ("all time:%.4f"%(time.time()-tic))
