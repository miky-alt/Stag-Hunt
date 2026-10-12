import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import math
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils
import tensorflow_probability as tfp


VALID_NETWORK_INITS = ('orthogonal', 'legacy')


def _orthogonal_layer(size, gain):
    """Builds a tanh dense layer with orthogonal weight initialization."""
    return tf.keras.layers.Dense(
        size,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=float(gain)),
        bias_initializer=tf.keras.initializers.Zeros(),
    )


# 'legacy' reproduces the June 2026 networks: ReLU + he_normal hidden, glorot_uniform output.
def _hidden_layer(size, init):
    """Builds a hidden layer using the selected initialization scheme."""
    if init == 'legacy':
        return tf.keras.layers.Dense(size, activation='relu', kernel_initializer='he_normal')
    return _orthogonal_layer(size, math.sqrt(2.0))


def _output_layer(size, init, orthogonal_gain):
    """Builds a linear output layer with the selected initialization."""
    if init == 'legacy':
        return tf.keras.layers.Dense(size, activation=None, kernel_initializer='glorot_uniform')
    return tf.keras.layers.Dense(
        size,
        activation=None,
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=orthogonal_gain),
        bias_initializer=tf.keras.initializers.Zeros(),
    )


def _validate_init(init):
    """Rejects network initialization modes that are not supported."""
    if init not in VALID_NETWORK_INITS:
        raise ValueError(f'network_init must be one of {VALID_NETWORK_INITS}, got {init!r}.')


class ActorNetwork(network.Network):
    """
    Neural network responsible for selecting actions (The Policy).
    Maps the flattened coordinate observation to a Categorical probability distribution.
    """
    def __init__(self, input_tensor_spec, output_tensor_spec, hidden_sizes=(128, 128), init='orthogonal'):
        """Creates the policy network for mapping observations to actions."""
        # Validate initialization and establish the TF-Agents network contract.
        _validate_init(init)
        # We must initialize the tf_agents Network superclass
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='ActorNetwork'
        )
        
        # Build hidden layers, then size the logits head from the action spec.
        self._output_tensor_spec = output_tensor_spec
        
        # Build the hidden layers dynamically
        self._hidden_layers = [_hidden_layer(size, init) for size in hidden_sizes]
            
        # The output layer matches the number of possible actions
        num_actions = output_tensor_spec.maximum - output_tensor_spec.minimum + 1
        self._action_logits = _output_layer(num_actions, init, orthogonal_gain=0.01)

    def call(self, inputs, step_type=None, network_state=(), training=False):
        """
        Forward pass. TF-Agents expects step_type and network_state arguments.
        """
        # Convert observations to the network's numeric type before inference.
        x = tf.cast(inputs, tf.float32)
        # tf.print("\n[Actor Network] Input shape (Batch_Size, Obs_Dim):", tf.shape(x))


        for layer in self._hidden_layers:
            x = layer(x, training=training)
            
        # Convert action logits into a categorical policy distribution.
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
    def __init__(self, input_tensor_spec, hidden_sizes=(128, 128), output_size=1, init='orthogonal'):
        """Creates a value network with the requested number of outputs."""
        # Validate the initializer and initialize the TF-Agents base network.
        _validate_init(init)
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name='CriticNetwork'
        )
        
        # Build the feature extractor and a linear value-prediction head.
        self._hidden_layers = [_hidden_layer(size, init) for size in hidden_sizes]
        self._output_size = output_size
            
        # Output layer is a single linear node
        self._value_output = _output_layer(output_size, init, orthogonal_gain=1.0)

    def call(self, inputs, step_type=None, network_state=(), training=False):
        """Predicts state values and returns the unchanged network state."""
        # Cast inputs, pass through hidden layers, and predict value outputs.
        x = tf.cast(inputs, tf.float32)
        # --- TENSORFLOW DEBUG LINE ---
        # tf.print captures runtime output from the device.

        for layer in self._hidden_layers:
            x = layer(x, training=training)
            
        value = self._value_output(x, training=training)
        
        # Keep one value per agent for centralized critics and scalar values
        # for local critics.
        if self._output_size == 1:
            value = tf.squeeze(value, axis=-1)
        
        return value, network_state