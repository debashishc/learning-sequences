


## Frequently asked questions

### Basic questions

What are activation functions, and why are they necessary?
Without an activation function like relu (also called a non-linearity), the Dense layer would consist of two linear operations—a dot product and an addition:

$$ \text{output = dot(W, input) + b}  $$
So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations, because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space.

In order to get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity, or activation function. relu is the most popular activation function in deep learning, but there are many other candidates, which all come with similarly strange names: prelu, elu, and so on.