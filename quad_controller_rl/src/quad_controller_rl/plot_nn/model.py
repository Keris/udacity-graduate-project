from keras import layers, models, optimizers
import keras.backend as K

class Actor:
    '''Actor (Policy) Model.'''

    def __init__(self, state_size, action_size, action_low, action_high, learning_rate=0.0001):
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
        self.learning_rate = learning_rate

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        '''Build an actor (policy) network that maps states -> actions.'''
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=256, activation='relu')(states)
        # net = layers.BatchNormalization()(net)
        # net = layers.Dropout(rate=0.3)(net)
        net = layers.Dense(units=128, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add batch normalization
        # net = layers.BatchNormalization()(net)
        # net = layers.Dropout(rate=0.3)(net)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    '''Critic (Value) Model.'''

    def __init__(self, state_size, action_size, learning_rate=0.001):
        '''Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        '''Build a critic (value) network that maps (state, action) pairs -> Q-values.'''
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=256, activation='relu')(states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Dropout(rate=0.2)(net_states)
        net_states = layers.Dense(units=128, activation='relu')(net_states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Dropout(rate=0.2)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=256, activation='relu')(actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Dropout(rate=0.2)(net_actions)
        net_actions = layers.Dense(units=128, activation='relu')(net_actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Dropout(rate=0.2)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss functions
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)


if __name__ == '__main__':
    actor = Actor(7, 3, 0, 10)
    critic = Critic(7, 3)
    from keras.utils import plot_model
    plot_model(actor.model, show_shapes=True, to_file='actor.png')
    plot_model(critic.model, show_shapes=True, to_file='critic.png')
