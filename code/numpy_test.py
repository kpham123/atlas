
def main1():
    import numpy as np
    # x = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
    x = np.ones((5,5))

    # Iterate over the matrix and remove top corners
    jis = []
    iis = []
    for i in range(0,int((x.shape[0]+1)/2)):
        for j in range(0,i):
            # Upper corner
            jis.append(j)
            iis.append(i)

            # Bottom corner
            jis.append(x.shape[0]-j-1)
            iis.append(i)
    x[np.array(jis),np.array(iis)] = 0

    # Zero out the matrix above half (this will be padding in the model)
    x[:,(x.shape[0]/2)+1:] = 0
    print(x)

    y = np.flip(x,axis=1)
    print(y)

    rot = np.rot90(x,k=1,axes=(1,0))
    print(rot)

    rot2 = np.rot90(x,k=1,axes=(0,1))
    print(rot2)

    z = x + y + rot + rot2
    print(z)

# for x in 
# print([(i,j) for j in range(0,i) for i in range(0,int((x.shape[0]+1)/2))])

def main2():
    import tensorflow as tf
    import numpy as np
    sess = tf.InteractiveSession()
    x = tf.ones((5,5,5),dtype=tf.float32)

    x = x[:,:,0:(x.shape[0]/2)+1]
    assert((x.get_shape().as_list()[0] == 5) and (x.get_shape().as_list()[1] == 5) and (x.get_shape().as_list()[2] == 3))

    # Clear pyramid along initial dimension 0
    
    # x = tf.matrix_set_diag(x,tf.zeros(tf.matrix_diag_part(x).shape))
    # print('x')
    # print(x.eval(0))
    # print(tf.reduce_sum(x).eval());exit()
    
    x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
    x = tf.pad(x,[[0,0],[0,0],[0,2]])
    x = tf.reverse(x,axis=[2])
    x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
    x = tf.reverse(x,axis=[2])

    # Clear pyramid along initial dimension 1
    x = tf.transpose(x,[1,0,2])
    x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
    x = tf.reverse(x,axis=[2])
    x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
    x = tf.reverse(x,axis=[2])
    x = tf.transpose(x,[1,0,2])

    # Rotate all pyramids back to their orientation in the 3-d volume
    x2 = tf.reverse(x,axis=[2])
    x3 = tf.transpose(x,[2,0,1])
    x4 = tf.reverse(x3,axis=[0])
    x5 = tf.transpose(x,[1,2,0])
    x6 = tf.reverse(x5,axis=[1])
    z = x + x2 + x3 + x4 + x5 + x6

    # Normalize overlapping contexts back to an equal weighting per voxel (i.e. along edges, diagonals, and center pixel of the cube, etc.)
    print('z')
    print(z.eval(0))
    print(tf.reduce_sum(z).eval())
main2()