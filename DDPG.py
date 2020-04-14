import tensorflow as tf
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
LR_Actor = 0.0001    # learning rate for actor
LR_Critic = 0.0001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01     # soft replacement
MEMORY_CAPACITY = 1500
BATCH_SIZE = 32

#####################  DDPG类  ####################
class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.memory = np.zeros((24,MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = np.zeros(24)
        self.sess = tf.Session()

        self.state_dim, self.action_dim = state_dim, action_dim
        self.State = tf.placeholder(tf.float32, [None, state_dim], 'State')
        self.State_ = tf.placeholder(tf.float32, [None, state_dim], 'State_')
        self.Reward = tf.placeholder(tf.float32, [None, 1], 'Reward')

        with tf.variable_scope('Actor'):
            self.action = self.build_actor(self.State, scope='eval', trainable=True)
            action_ = self.build_actor(self.State_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            Q = self.build_critic(self.State, self.action, scope='eval', trainable=True)
            Q_ = self.build_critic(self.State_, action_, scope='target', trainable=False)

        self.Actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.Actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.Critic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.Critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [[tf.assign(Atp, (1 - TAU) * Atp + TAU * Aep), tf.assign(Ctp, (1 - TAU) * Ctp + TAU * Cep)]
                                     for Atp, Aep, Ctp, Cep in zip(self.Actor_target_params, self.Actor_eval_params,
                                                                   self.Critic_target_params, self.Critic_eval_params)]



        Q_target = self.Reward + GAMMA * Q_

        self.loss_critic = tf.reduce_mean(tf.square(Q_target - Q))
        self.train_critic = tf.train.AdamOptimizer(LR_Critic,epsilon=1e-08).minimize(self.loss_critic, var_list=self.Critic_eval_params)

        self.loss_actor = -tf.reduce_mean(Q)  # maximize the q
        self.train_actor = tf.train.AdamOptimizer(LR_Actor).minimize(self.loss_actor, var_list=self.Actor_eval_params)

        self.sess.run(tf.global_variables_initializer())

        self.cost_critic = []
        self.cost_actor = []

    ##################### Actor网络构建 ################################
    def build_actor(self, State, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(State, 512, activation=tf.nn.relu, name='l1', trainable=trainable)
            net1 = tf.layers.dense(net, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net1, self.action_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    ##################### Critic网络构建 ################################
    def build_critic(self, State, action, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.tanh(tf.matmul(State, w1_s) + tf.matmul(action, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    # 产生下一个动作
    def choose_action(self, state):
        return self.sess.run(self.action, feed_dict={self.State: state})

    # 记忆池建立
    def store_tracsition(self, State, action, reward, State_, t):

        transition = np.hstack((State, action, [[reward]], State_))
        index = self.pointer[t] % MEMORY_CAPACITY
        index = int(index)
        self.memory[t, index, :] = transition
        self.pointer[t] += 1

    def learn(self,t):
        self.sess.run(self.soft_replace)
        index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        buffer = self.memory[t,index, :]
        buffer_state = buffer[:, :self.state_dim]
        buffer_action = buffer[:, self.state_dim: self.state_dim + self.action_dim]  # d_dim：action的维度
        buffer_reward = buffer[:, -self.state_dim - 1: -self.state_dim]
        buffer_state_ = buffer[:, -self.state_dim:]
        _, loss_actor = self.sess.run([self.train_actor, self.loss_actor],
                                 feed_dict={self.State: buffer_state})
        _, loss_critic = self.sess.run([self.train_critic, self.loss_critic],
                                  feed_dict={self.State: buffer_state, self.action: buffer_action,
                                             self.Reward: buffer_reward, self.State_: buffer_state_,
                                             })
        self.cost_critic.append(loss_critic)
        self.cost_actor.append(loss_actor)
    # 测试
    def test(self):
        s = [[0.0, 0.0]]
        s_ = [[0.0, 1.0]]
        R = [[1.0]]
        a = [[0.5, 0.3332222]]
        print("**********************")
        print(self.sess.run([self.action, self.action_], feed_dict={self.State: s, self.State_: s_, self.Reward: R}))
        print("**********************")
        print(self.sess.run(self.Actor_eval_params))
        print(self.sess.run(self.Actor_target_params))
        print(self.memory[0])

    def plot_cost(self):
        print("cost_critic:")
        print(self.cost_critic)
        plt.figure()
        plt.plot(self.cost_critic, label="Critic_error")
        plt.ylabel('Q_Loss')
        plt.xlabel('training steps')
        # plt.savefig("../model/QLoss2.png")
        plt.show()

        print("cost_actor:")
        print(self.cost_actor)
        plt.figure()
        plt.plot(self.cost_actor, label="Actor_error")
        plt.ylabel('Action')
        plt.xlabel('training steps')
        # plt.savefig("../model/QLoss2.png")
        plt.show()



if __name__ == "__main__":
    #  测试
    W  = DDPG(2, 2)
    s = [[1.0, 0.0]]
    a = [[0.4, 0.222222]]
    R = 1.0
    print(W.choose_action(s))
