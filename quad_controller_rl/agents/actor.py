import tensorflow as tf


class Actor:
    '''Actor (Policy) Model.'''

    def __init__(self, state_size, action_size, action_low, action_high):
        '''Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        print('state_size: {}, action_size: {}'.format(self.state_size, self.action_size))
        print('action_range: {}'.format(self.action_range))

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        '''Build an actor (policy) network that maps states -> actions.'''
        # Define input layer (states)
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = tf.keras.layers.Dense(units=32, activation='relu')(states)
        net = tf.keras.layers.Dense(units=64, activation='relu')(net)
        net = tf.keras.layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = tf.keras.layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = tf.keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = tf.keras.models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = tf.keras.layers.Input(shape=(self.action_size,))
        loss = tf.keras.backend.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = tf.keras.optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = tf.keras.backend.function(
            inputs=[self.model.input, action_gradients, tf.keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)

