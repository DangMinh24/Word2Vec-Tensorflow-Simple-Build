import numpy as np

vocabulary=["apple","eat","juice","drink","milk","orange","rice","water"]
vocabulary_size=len(vocabulary)

train_data=[("eat","apple"),("eat","orange"),("eat","rice"),("drink","milk"),
            ("drink","juice"),("drink","water"),("orange","juice"),("apple","juice"),
            ("rice","milk"),("milk","drink"),("water","drink"),("juice","drink")]

dictionary={}
index=0
for word in vocabulary:
    dictionary[word]=index
    index+=1

reversed_dic=dict(zip(dictionary.values(),dictionary.keys()))
print(reversed_dic.items())

count={}

import tensorflow as tf
num_hidden_nodes=3
lr=0.1

sess=tf.InteractiveSession()
input=tf.placeholder(tf.int32,shape=[1,vocabulary_size])

label=tf.placeholder(tf.int32,shape=[1])

W=tf.Variable(tf.random_uniform([vocabulary_size,num_hidden_nodes],-1.0,1.0,seed=5))

W2=tf.Variable(tf.random_uniform([num_hidden_nodes,vocabulary_size],-1.0,1.0,seed=1))

z1=tf.matmul(tf.to_float(input),W)

z2=tf.matmul(z1,W2)

exp_z2=tf.exp(z2)
a2=tf.div(exp_z2,tf.reduce_sum(exp_z2))

init=tf.initialize_all_variables()

sess.run(init)

tmp_instance=train_data[0]
def pair2instance(pair):
    input_word,label=pair[0],pair[1]
    input_vector=np.zeros(vocabulary_size)
    input_vector[dictionary[input_word]]=1

    label=dictionary[label]
    return [input_vector],[label]
input_vector,label_in=pair2instance(tmp_instance)
feed={input:input_vector,label:label_in}

pos=z2[0][label_in]
loss=tf.log(tf.reduce_sum(exp_z2))-pos
dz2=tf.gradients(loss,z2)[0]
dw2=tf.gradients(loss,W2)[0]
W2_new=tf.sub(W2,tf.mul(dw2,lr))

dw=tf.gradients(loss,W)[0]
W_new=tf.sub(W,tf.mul(dw,lr))

# update_2=tf.assign(W2,W2_new)
# update_1=tf.assign(W,W_new)

iter=500
for i in range(iter):
    print("loss: ",sess.run(loss,feed_dict=feed))
    W2_new_=sess.run(W2_new,feed_dict=feed)
    W_new_=sess.run(W_new,feed_dict=feed)
    update_1=tf.assign(W,W_new_)
    update_2=tf.assign(W2,W2_new_)
    sess.run(update_1)
    sess.run(update_2)
    # print(sess.run(W2))
    print()


