
import tensorflow as tf
from tensorflow.keras.layers import Dense   # type: ignore   

class Attention(tf.keras.layers.Layer):
    """
    Multi-head Attention Layer implementation.
    This layer implements multi-head attention mechanism as described in 
    "Attention Is All You Need" (Vaswani et al., 2017).
    It projects the input into query, key and value vectors, splits them into multiple heads,
    applies scaled dot-product attention, and combines the results.
    Args:
        embed_dim (int): The embedding dimension
        num_heads (int, optional): Number of attention heads. Defaults to 8
        **kwargs: Additional keyword arguments passed to the parent Layer class
    Attributes:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        projection_dim (int): Dimension of each attention head
        query_dense (Dense): Linear transformation for query
        key_dense (Dense): Linear transformation for key  
        value_dense (Dense): Linear transformation for value
        combine_heads (Dense): Linear transformation to combine attention heads
    Methods:
        - attention(query, key, value): Computes scaled dot-product attention
        - separate_heads(x, batch_size): Splits input into multiple attention heads
        - call(inputs): Main method to compute multi-head attention
        - get_config(): Returns layer configuration
    Raises:
        ValueError: If embedding dimension is not divisible by number of heads
    Returns:
        tf.Tensor: Output tensor after applying multi-head attention
    References:
        Vaswani, A., et al. (2017). Attention Is All You Need. arXiv:1706.03762
    """

    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        """
        Implements the scaled dot-product attention mechanism.
        This function calculates attention scores between query and key vectors, scales them,
        applies softmax to get attention weights, and computes weighted sum with values.
        The formula is: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        Args:
            query: Tensor of shape (..., seq_len_q, depth)
                containing the query vectors
            key: Tensor of shape (..., seq_len_k, depth)
                containing the key vectors  
            value: Tensor of shape (..., seq_len_v, depth_v)
                containing the value vectors
        Returns:
            tuple:
                - output: Tensor of shape (..., seq_len_q, depth_v)
                    containing the computed attention values
                - weights: Tensor of shape (..., seq_len_q, seq_len_k)
                    containing the attention weights
        Note: seq_len_k and seq_len_v must be equal
        """
        
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)

        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Separates the input tensor into multiple attention heads.
        This function reshapes the input tensor and transposes it to separate the attention heads
        for parallel processing in the multi-head attention mechanism.
        Args:
            x (tf.Tensor): Input tensor to be separated into heads.
                Shape: (batch_size, seq_length, num_heads * projection_dim)
            batch_size (int): Size of the batch dimension.
        Returns:
            tf.Tensor: Tensor with separated attention heads.
                Shape: (batch_size, num_heads, seq_length, projection_dim)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Applies multi-head self attention mechanism on input tensor.
        This method implements the core attention mechanism by:
        1. Creating query, key and value tensors through dense layers
        2. Separating the heads
        3. Computing scaled dot-product attention
        4. Combining the heads back
        5. Returning the final output
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            output: Tensor of shape (batch_size, seq_len, embed_dim) after applying
                multi-head attention mechanism
        Notes:
            - The input is projected into query, key and value spaces
            - Attention is computed in parallel for num_heads different representation subspaces
            - The attention scores are computed as scaled dot product between query and key
            - The output combines information from all attention heads
        """

        # Get batch size from inputs
        batch_size = tf.shape(inputs)[0]
        
        # Project inputs to query, key and value spaces
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Compute attention
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output
    
    def get_config(self):
        """
        Gets the configuration of the Multi-Head Self-Attention module.
        Returns:
            dict: A dictionary containing the configuration parameters, including:
                - embed_dim (int): The embedding dimension
                - num_heads (int): Number of attention heads
                - All config parameters from parent class
        """
        
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,
                       "num_heads": self.num_heads,})
        
        return config