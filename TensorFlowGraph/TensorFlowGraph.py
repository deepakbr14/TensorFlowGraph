############################################################################################################
#
# i used this webpage and retyped/recoded it below:
#   https://d3lm.medium.com/understand-tensorflow-by-mimicking-its-api-from-scratch-faa55787170d
#
# NOTE: i'm using tensorflow 2 for this
#
# tensor flow built of two parts
#
#   a)  library for defining computational graphs
#
#       computational graph =   abstract way of describing computations as a directed graph
#                               (aka data flow graphs)
#
#       directed graph      =   data structure consisting of nodes and edges
#                               see https://miro.medium.com/max/300/1*V6aYjD3AxDbEKYahkGqVQw.png
#
#                               you can use graphs to represent things like 
#                                   road networks
#                                   social networks
#                                   neural networks
#                                       (meaning tensor flow isn't necessarily a neural network library.
#                                        it's a directed graph library - which happens to be the underlying
#                                        structure of a neural net - which is why it works well for neural
#                                        networks)
#
#                               nodes can be
#                                   operations (ops)    =   create or manipulate data according to specific rules
#                                   variables           =   shared/presistent data that can be manipulated by ops. must be initilized
#                                   placeholders        =   for things like input that don't need to be initilized
#                                   constants           =   parameters that cannot be changed
#
#                                  Q:  why are variables intialized and placeholders not?
#                                     
#                                      imagine you are building a neural net for me.
#                                      i need your neural net to have all of its weights and biases set (variables)
#                                      but i don't need your input data, since i'm going to run it with my own (placeholders)
#
#                               edges = data (multidimensonal array aka tensors)
#                                       edges carry info from one node to the other
#
#           the tensorflow object for this is called Graph
#
#   b)  runtime for executing your graphs on hardware
#   
#           the tensorflow object for this is called Session
#
#           you use a Session to create a runtime
#               this bascially takes your whole data flow (with all of its nodes/edges)
#               and sets is up to run on your cpu/gpu/tpus etc 
#               it allocates all the needed memory etc
#


#standard program
a = 15;
b = 5;
prod = a * b
sum = a + b
res = prod / sum
print("standard program result is", res)

#directed graph version of the same program
#see https://miro.medium.com/max/700/1*vPb9E0Yd1QUAD0oFmAgaOw.png

#NOTE:  you don't need to think of graphs as left to right
#       they could be top down, bottom up, forever looping, whatever

#Q: why organize the program/computation like this?
#   one benefit is the graph version of the program shows that
#       the prod (multiplication) and the sum can be calculated at the same time
#       (since they don't depend on each other)
#   just by looking at the graph, the runtime can send the prod to one
#       processor and the sum to another
#   this is not obvious when looking at the standard program


#creating the Graph and the Session
#   when creating the Graph in code, the actual memory is not allocated until
#   the Session is created.
#    

#this is our standard program rewritten in tensorflow
import tensorflow as tf

#NOTE:  using 'with' is apparently a shorthand way of try/catch block
#       it automatically closes things for us etc.
#       am not clear on exacly how this works - ie how/where do the objects specify what things needs to closed?

#with tf.Session() as sess:
#NOTE:  because i'm using tensorflow 2 i have to use 
#           'tf.compat.v1.Session() as sess:'
#       but if i were using tensorflow 1 it would just look like with 
#           tf.Session() as sess:
with tf.compat.v1.Session() as sess:

    #create the Graph

    #NOTE:  at this point tensorflow has already created a 'default graph'
    #       not sure why they did this.  apparently most people just use the default
    #       but we could create our own?

    #this creates constant with the name 'a'
    #each time we create something, it automatically gets added to the graph
    #so the shape of the graph will be generated based on your statements
    #   i.e.    tf.divide uses prod and sum as inputs - so prod and sum
    #           would automatically have to sit to the left of the res

    a       = tf.constant(15, name = "a");
    b       = tf.constant(5,  name = "b");

    prod    = tf.multiply(a, b, name = "Multiply");
    sum     = tf.add(a, b, name = "Add");
    res     = tf.divide(prod, sum, name = "Divide");

    #at this point the Graph has been created, but nothing has been
    #allocated\run

    #run the Session
    #we only need to pass res to the run because sess.run it will automatically figure out all the dependencies.
    #   i.e.    to process res i need prod and sum
    #           but to process prod i need a and b
    out = sess.run(res);

    print("tensorflow program result is", out)


#NOTE:  when i run this i see
#       standard program result is 3.75
#       bunch of junk telling how unoptimized/crappy my tensorflow setup is
#       tensorflow program result is 3.75

