import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils
import tensorflow_probability as tfp

class ActorNetwork(network.Network):
    """
    Neural network responsible for selecting actions (The Policy).
    Maps the flattened coordinate observation to a Categorical probability distribution.
    """
    def __init__(self, input_tensor_spec, output_tensor_spec, hidden_sizes=(128, 128)):
        # We must initialize the tf_agents Network superclass
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='ActorNetwork'
        )
        
        self._output_tensor_spec = output_tensor_spec
        
        # Build the hidden layers dynamically
        self._hidden_layers = [
            tf.keras.layers.Dense(size, activation='relu', kernel_initializer='he_normal')
            for size in hidden_sizes
        ]
            
        # The output layer matches the number of possible actions
        num_actions = output_tensor_spec.maximum - output_tensor_spec.minimum + 1
        self._action_logits = tf.keras.layers.Dense(
            num_actions,
            activation=None, # Logits should be linear
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs, step_type=None, network_state=(), training=False):
        """
        Forward pass. TF-Agents expects step_type and network_state arguments.
        """
        # Ensure inputs are correctly batched
        x = tf.cast(inputs, tf.float32)
        # tf.print("\n[Actor Network] Input shape (Batch_Size, Obs_Dim):", tf.shape(x))


        for layer in self._hidden_layers:
            x = layer(x, training=training)
            
        logits = self._action_logits(x, training=training)
        # --- LOGIT DEBUG LINE ---
        # tf.print("[Actor Network] Generated logits (2x5 matrix):", logits)
        # --------------------------------
        # PPO requires a probability distribution, not just argmax
        action_distribution = tfp.distributions.Categorical(logits=logits)
        
        return action_distribution, network_state


class CriticNetwork(network.Network):
    """
    Neural network responsible for evaluating the state (The Value Function).
    Maps the flattened coordinate observation to a single expected return value.
    """
    def __init__(self, input_tensor_spec, hidden_sizes=(128, 128)):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='CriticNetwork'
        )
        
        self._hidden_layers = [
            tf.keras.layers.Dense(size, activation='relu', kernel_initializer='he_normal')
            for size in hidden_sizes
        ]
            
        # Output layer is a single linear node
        self._value_output = tf.keras.layers.Dense(
            1, 
            activation=None,
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs, step_type=None, network_state=(), training=False):
        x = tf.cast(inputs, tf.float32)
        # --- TENSORFLOW DEBUG LINE ---
        # tf.print captures runtime output from the device.

        for layer in self._hidden_layers:
            x = layer(x, training=training)
            
        value = self._value_output(x, training=training)
        
        # The critic must return the value as a tuple (outputs, state).
        # Remove the dimension added by Dense(1).
        value = tf.squeeze(value, axis=-1)
        
        return value, network_state