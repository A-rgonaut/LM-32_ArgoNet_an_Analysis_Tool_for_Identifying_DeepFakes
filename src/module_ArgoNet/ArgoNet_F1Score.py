
import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    """
    A custom F1 Score metric implementation for TensorFlow/Keras models.
    This class implements the F1 Score metric, which is the harmonic mean of precision and recall.
    It handles binary classification tasks and maintains running counts of true positives (TP),
    false positives (FP), and false negatives (FN) to compute the metric.
    Args:
        name (str, optional): Name of the metric. Defaults to 'f1_score'.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    Attributes:
        threshold (float): Classification threshold for binary predictions.
        tp (tf.Variable): Running count of true positives.
        fp (tf.Variable): Running count of false positives.
        fn (tf.Variable): Running count of false negatives.
    Methods:
        - update_state(y_true, y_pred, sample_weight=None): Updates the state variables used to compute the F1 score.
        - result(): Computes and returns the F1 score based on current state.
        - reset_states(): Resets all state variables back to zero.
    Example:
        ```python
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=[F1Score()])
        ```
    """

    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state variables for F1 score calculation.
        Args:
            y_true (tensor): Ground truth labels.
            y_pred (tensor): Predicted labels (raw probabilities).
            sample_weight (tensor, optional): Sample weights. Defaults to None.
        Notes:
            - Converts inputs to float32 
            - Expands y_true dimensions if needed to match y_pred
            - Thresholds y_pred to binary values
            - Computes and accumulates:
                - True Positives (TP)
                - False Positives (FP) 
                - False Negatives (FN)
        Updates:
            self.tp: Accumulated true positives
            self.fp: Accumulated false positives 
            self.fn: Accumulated false negatives
        """

        # Convert y_true and y_pred to float32 and ensure shapes match
        y_true = tf.keras.backend.cast(y_true, 'float32')
        if len(y_true.shape) < len(y_pred.shape):
            y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.keras.backend.cast(y_pred > self.threshold, 'float32')

        # Count TP, FP, FN
        tp = tf.keras.backend.sum(y_true * y_pred)
        fp = tf.keras.backend.sum((1 - y_true) * y_pred)
        fn = tf.keras.backend.sum(y_true * (1 - y_pred))

        # Update the state variables
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        """
        Calculate the F1 score based on true positives (tp), false positives (fp), and false negatives (fn).
        The F1 score is the harmonic mean of precision and recall, providing a single score that balances both metrics.
        It ranges from 0 (worst) to 1 (best).
        Returns:
            float: The F1 score computed as 2 * (precision * recall) / (precision + recall).
                  Uses epsilon in denominator to avoid division by zero.
        """

        # Compute precision, recall, and F1 score
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        """
        Resets all state variables used for F1 score calculation to zero.
        This method resets the true positive (tp), false positive (fp), and false negative (fn)
        counters back to their initial state of 0. This is typically called at the start of
        each training epoch or evaluation phase.
        """

        # Reset the state variables to zero
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
