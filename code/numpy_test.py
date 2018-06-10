
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
    x = tf.ones((f5,5))

    x = x[:,0:(x.shape[0]/2)+1]
    x = tf.matrix_band_part(x,-1,0)
    x = tf.pad(x,[[0,0],[0,2]])
    x = tf.reverse(x,axis=[1])
    x = tf.matrix_band_part(x,0,-1)
    x = tf.reverse(x,axis=[1])

    print(x.eval())

main2()