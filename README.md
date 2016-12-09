# Word2Vec-Tensorflow-Simple-Build

Word2Vec is a very important step for semantic analysis and for many tasks in NLP. 
Word2Vec is a function that try to solve an interesting problem: represent a word that machine can understand somehow about meaning. Normally, we can use binary code like 10101101... to express data, we can apply that to word also. However, it can't express a lot of semantic infomation.

Another solution to express word is a vector with a lot of dimensions (k-dimensions). Each words has an unique vector, also has an unique value. These values somehow have a relations between them. 

Example: ("man")+("young")=("boy"), ("king")-("man")=("queen")  

There are lots of tutorial to help you to understand about __how word embeddings work__:

1/[Vector Representations of Words](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html) of Tensorflow :

2/[wevi: word embedding visual inspector](https://ronxin.github.io/wevi/) by [Xin Rong](http://www-personal.umich.edu/~ronxin/) and his explaination in [the paper](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf) 

However, I will try to rebuild a word2vec in a most basic way. Although that Tensorflow has already explained the basic idea, I think that [the code in tensorflow](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) still be a blackbox to beginner (like me!).

The basic idea about word embeddings is using Neural Network to train a model where input is a context word and output is a target word (You can go to [wevi](https://ronxin.github.io/wevi/), train 100 iter, than see that if input is "drink", then output will be "juice", "milk", "water")

So, I will try to make a Neural Network model, with input is a word(context word) and output is also a word(target word) by using Tensorflow (which very useful for Neural NetWork)

    input=tf.placeholder(tf.int32,shape=[1,vocabulary_size])
    num_hidden_nodes=3
    W=tf.Variable(tf.random_uniform([vocabulary_size,num_hidden_nodes],-1.0,1.0,seed=5))
    W2=tf.Variable(tf.random_uniform([num_hidden_nodes,vocabulary_size],-1.0,1.0,seed=1))
    z1=tf.matmul(tf.to_float(input),W)
    z2=tf.matmul(z1,W2
Input is an one-hot encoding vector for context word, num_hidden_nodes to determine how many nodes in hidden layers. W matrix to represent weight input->hidden layers, W2 matrix to represent weight hidden->output layer
z1 is a result of multiplication (input\*W).
Usually there will be a activation function of each layer (hiddens,output). To simplify, in hidden layer, we use linear function, which simply keep the same result (input\*W).Similar to z2. 

According to 1/, a function that need to maximize is: J=z2_t - log(sum(exp(z2))) => loss=log(sum(exp(z2))) - z2_t where z2_t is a z2 of t where t=target word

To minimize loss function, I use Gradient Descent Optimization, which finds derivation of W and W2, then update W and W2 according to formular: new_W= old_W -learning_rate\*derivation

