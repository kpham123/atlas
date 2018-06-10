
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
    """ Version without hidden units. """
    import tensorflow as tf
    import numpy as np
    sess = tf.InteractiveSession()
    batchNumber = 2
    x = tf.ones((batchNumber,5,5,5),dtype=tf.float32)

    x = x[:,:,:,0:(x.shape[3]/2)+1]
    # assert((x.get_shape().as_list()[0] == 5) and (x.get_shape().as_list()[1] == 5) and (x.get_shape().as_list()[2] == 3))

    # Pad back to cubic
    x = tf.pad(x,[[0,0],[0,0],[0,0],[0,2]])

    def aggregate(x):
        def clean(x):
            # Clear pyramid along initial dimension 0
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[3])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[3])

            # Clear pyramid along initial dimension 1
            x = tf.transpose(x,[0,2,1,3])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[3])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[3])
            x = tf.transpose(x,[0,2,1,3])
            return x

        # Rotate all pyramids back to their orientation in the 3-d volume
        x1 = clean(x)
        x2 = tf.reverse(x1,axis=[3])
        x3 = tf.transpose(x1,[0,3,1,2])
        x4 = tf.reverse(x3,axis=[1])
        x5 = tf.transpose(x1,[0,2,3,1])
        x6 = tf.reverse(x5,axis=[2])
        z = x1 + x2 + x3 + x4 + x5 + x6
        return z

    # Normalize overlapping contexts back to an equal weighting per voxel (i.e. along edges, diagonals, and center pixel of the cube, etc.)
    #  Generate the matrix to divide element-wise
    z = aggregate(x)
    print('z')
    print(z[0].eval(0))
    print(tf.reduce_sum(z).eval())
    
    p_normalization = aggregate(tf.ones(x.shape))
    z = tf.divide(z,p_normalization)
    print('z')
    print(z[0].eval(0))
    print(tf.reduce_sum(z).eval())

def main2_with_hidden():
    """ Version with hidden unit dimension. """
    import tensorflow as tf
    import numpy as np
    sess = tf.InteractiveSession()
    batchNumber = 2
    x = tf.ones((batchNumber,5,5,5,1),dtype=tf.float32)

    x = x[:,:,:,0:(x.shape[3]/2)+1,:]
    # assert((x.get_shape().as_list()[0] == 5) and (x.get_shape().as_list()[1] == 5) and (x.get_shape().as_list()[2] == 3))

    # Pad back to cubic
    x = tf.pad(x,[[0,0],[0,0],[0,0],[0,2],[0,0]])

    def aggregate(x):
        def clean(x):
            # Temporarily transpose hidden units to dimension 1
            x = tf.transpose(x,[0,4,1,2,3])

            # Clear pyramid along initial dimension 0
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[4])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[4])

            # Clear pyramid along initial dimension 1
            x = tf.transpose(x,[0,1,3,2,4])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[4])
            x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
            x = tf.reverse(x,axis=[4])
            x = tf.transpose(x,[0,1,3,2,4])

            # Undo temporary tranpose of hidden units
            x = tf.transpose(x,[0,2,3,4,1])
            return x

        # Rotate all pyramids back to their orientation in the 3-d volume
        x1 = clean(x)
        x2 = tf.reverse(x1,axis=[3])
        x3 = tf.transpose(x1,[0,3,1,2,4])
        x4 = tf.reverse(x3,axis=[1])
        x5 = tf.transpose(x1,[0,2,3,1,4])
        x6 = tf.reverse(x5,axis=[2])
        z = x1 + x2 + x3 + x4 + x5 + x6
        return z

    z = aggregate(x)
    print('z')
    print(z[0,:,:,:,0].eval(0))
    print(tf.reduce_sum(z).eval())
    
    p_normalization = aggregate(tf.ones(x.shape))
    z = tf.divide(z,p_normalization)
    print('z')
    print(z[0,:,:,:,0].eval(0))
    print(tf.reduce_sum(z).eval())

def tt():
    """ Transpose test """
    import tensorflow as tf
    import numpy as np
    sess = tf.InteractiveSession()
    cube = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
    print('cube:',cube)
    norm = np.array([[[3,2,3],[2,1,2],[3,2,3]],[[2,1,2],[1,6,1],[2,1,2]],[[3,2,3],[2,1,2],[3,2,3]]])
    norm = tf.constant(norm,dtype=tf.float32)
    x = tf.constant(cube,dtype=tf.float32)
    # x = tf.ones(cube.shape)
    x = tf.expand_dims(x,-1)
    x = tf.expand_dims(x,0)
    print(x.get_shape().as_list())

    maxSize = 3
    time_index_length = 3
    a = tf.transpose(x,np.array([0,1,2,3,4]))
    a = a[:,0:int((time_index_length+1)/2),:,:,:]
    # print('a:',a.get_shape().as_list())
    # print(tf.squeeze(a,[0,4]).eval())

    b = tf.transpose(x,np.array([0,1,2,3,4]))
    b = tf.reverse(b[:,int((time_index_length-1)/2):,:,:,:],axis=[1])
    # print('b:',a.get_shape().as_list())
    # print(tf.squeeze(b,[0,4]).eval());exit()

    c = tf.transpose(x,np.array([0,2,3,1,4]))
    c = c[:,0:int((time_index_length+1)/2),:,:,:]
    d = tf.transpose(x,np.array([0,2,3,1,4]))
    d = tf.reverse(d[:,int((time_index_length-1)/2):,:,:,:],axis=[1])

    e = tf.transpose(x,np.array([0,3,1,2,4]))
    e = e[:,0:int((time_index_length+1)/2),:,:,:]
    f = tf.transpose(x,np.array([0,3,1,2,4]))
    f = tf.reverse(f[:,int((time_index_length-1)/2):,:,:,:],axis=[1])

    # for idx,clstm in enumerate([a,b,c,d,e,f]):
    #   clstm = tf.transpose(clstm,[0,2,3,1,4])
    #   clstm = tf.pad(clstm,[[0,0],[0,0],[0,0],[0,maxSize-clstm.shape[3].value],[0,0]])
    #   assert((clstm.get_shape().as_list()[1] == 3) and (clstm.get_shape().as_list()[2] == 3) and (clstm.get_shape().as_list()[3] == 3))
    #   print('{}:{}'.format(idx,clstm.get_shape().as_list()))
    # assert((a.get_shape().as_list()[1] == 3) and (a.get_shape().as_list()[2] == 3) and (a.get_shape().as_list()[3] == 3))
    a = tf.transpose(a,[0,2,3,1,4])
    a = tf.pad(a,[[0,0],[0,0],[0,0],[0,maxSize-a.shape[3].value],[0,0]])
    assert((a.get_shape().as_list()[1] == 3) and (a.get_shape().as_list()[2] == 3) and (a.get_shape().as_list()[3] == 3))

    b = tf.transpose(b,[0,2,3,1,4])
    b = tf.pad(b,[[0,0],[0,0],[0,0],[0,maxSize-b.shape[3].value],[0,0]])
    c = tf.transpose(c,[0,2,3,1,4])
    c = tf.pad(c,[[0,0],[0,0],[0,0],[0,maxSize-c.shape[3].value],[0,0]])
    d = tf.transpose(d,[0,2,3,1,4])
    d = tf.pad(d,[[0,0],[0,0],[0,0],[0,maxSize-d.shape[3].value],[0,0]])
    e = tf.transpose(e,[0,2,3,1,4])
    e = tf.pad(e,[[0,0],[0,0],[0,0],[0,maxSize-e.shape[3].value],[0,0]])
    f = tf.transpose(f,[0,2,3,1,4])
    f = tf.pad(f,[[0,0],[0,0],[0,0],[0,maxSize-f.shape[3].value],[0,0]])

    def clean(x):
        # Temporarily transpose hidden units to dimension 1
        x = tf.transpose(x,[0,4,1,2,3])

        print(x.get_shape().as_list())
        # Clear pyramid along initial dimension 0
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[4])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[4])

        # Clear pyramid along initial dimension 1
        x = tf.transpose(x,[0,1,3,2,4])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,-1,0),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[4])
        x = tf.map_fn(lambda z:tf.matrix_band_part(z,0,-1),x,parallel_iterations=x.get_shape().as_list()[0])
        x = tf.reverse(x,axis=[4])
        x = tf.transpose(x,[0,1,3,2,4])

        # Undo temporary tranpose of hidden units
        x = tf.transpose(x,[0,2,3,4,1])
        return x

    clstm1 = clean(a)
    print('clstm1:',clstm1.get_shape().as_list())
    print(tf.squeeze(clstm1,[0,4]).eval())

    clstm2 = clean(b)
    clstm2 = tf.reverse(clstm2,axis=[3])
    print('clstm2:',clstm2.get_shape().as_list())
    print(tf.squeeze(clstm2,[0,4]).eval())

    clstm3 = clean(c)
    clstm3 = tf.transpose(clstm3,[0,3,1,2,4])
    print('clstm3:',clstm3.get_shape().as_list())
    print(tf.squeeze(clstm3,[0,4]).eval())

    clstm4 = clean(d)
    clstm4 = tf.transpose(clstm4,[0,3,1,2,4])
    clstm4 = tf.reverse(clstm4,axis=[1])
    print('clstm4:',clstm4.get_shape().as_list())
    print(tf.squeeze(clstm4,[0,4]).eval())

    clstm5 = clean(e)
    clstm5 = tf.transpose(clstm5,[0,2,3,1,4])
    print('clstm5:',clstm5.get_shape().as_list())
    print(tf.squeeze(clstm5,[0,4]).eval())

    clstm6 = clean(f)
    clstm6 = tf.transpose(clstm6,[0,2,3,1,4])
    clstm6 = tf.reverse(clstm6,axis=[2])
    print('clstm6:',clstm6.get_shape().as_list())
    print(tf.squeeze(clstm6,[0,4]).eval())

    z = clstm1+clstm2+clstm3+clstm4+clstm5+clstm6
    print('z:',z.get_shape().as_list())
    z = tf.squeeze(z,[0,4])
    z = tf.divide(z,norm)
    print(z.eval())

    z = tf.transpose(z,[2,0,1])
    print('z_prime:',z.get_shape().as_list())
    print(z.eval())
    

# main2_with_hidden()
tt()