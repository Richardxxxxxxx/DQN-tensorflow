from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
from .ops import linear, conv2d, clipped_error, RNN, calculate_linear, define_linear
from .utils import get_time, save_pkl, load_pkl
#from tensorflow.python.profiler import model_analyzer
#from tensorflow.python.profiler import option_builder

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()
    #self.profiler = model_analyzer.Profiler(self.sess.graph)
        

  def train(self):
  
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      t1 = time.time()
      action = self.predict(self.history.get())
      # 2. act
      t2 = time.time()
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      t3 = time.time()
      self.observe(screen, reward, action, terminal)
      t4 = time.time()
      if self.step >= self.learn_start:
        #if self.step % self.test_step == self.test_step - 1:
        if self.step % (self.train_frequency*self.scale) == 0:
            print('\npre: %f, act: %f, obs: %f' % (t2-t1, t3-t2, t4-t3))

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()
        for _ in range(self.history_length):
          self.history.add(screen)

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1})
            self.save_model(self.step + 1)

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

#  def predict(self, s_t, action_plus_1 = [0], test_ep=None):
  def predict(self, s_t, test_ep=None):

    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      #action = self.q_action.eval({self.s_t: [s_t], self.action_plus: [action_plus_1]})[0]
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_ts, actions, rewards, terminal = self.memory.sample()

    t = time.time()
    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval({self.s_t: s_ts[1]})

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_ts[1],
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      })
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + rewards[0]
    else:
      #q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1, self.target_action_plus : [actions_plus_1]})
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_ts[1]})

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + rewards[0]

      target_q_g_ts = []
      #q_t_plus_2 = self.target_q.eval({self.target_s_t: s_t_plus_2, self.target_action_plus : [actions_plus_1]})
      for s_t_plus, reward_plus_deduct_1 in zip(s_ts[2:],rewards[1:]):   
          q_t_plus = self.target_q.eval({self.target_s_t: s_t_plus})
          max_q_t_plus = np.max(q_t_plus, axis=1)
          target_q_g_t = (1. - terminal) * self.discount * max_q_t_plus + reward_plus_deduct_1
          target_q_g_ts.append(target_q_g_t)

    #for op, self.action_plus shouble actions [1:]
    #for self.q_g, self.action_sequence should use actions [0:-1]
    #run_metadata = tf.RunMetadata()
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    _, q_t, q_g_t, loss, summary_str = self.sess.run([self.optim, self.q, self.q_g, self.loss, self.q_summary], {
      self.s_t: s_ts[0],
      self.target_q_t: target_q_t,
      self.action: actions[0],
      self.target_q_g_ts: target_q_g_ts,
      self.action_plus: actions[1:],
      self.action_sequence: actions[:-1],
      self.learning_rate_step: self.step,
    })
    #self.profiler.add_step(self.step, run_metadata)
    #profile_op_opt_builder = option_builder.ProfileOptionBuilder()
    #profile_op_opt_builder.select(['micros','occurrence'])
    #profile_op_opt_builder.order_by('micros')
    #profile_op_opt_builder.with_max_depth(4)
    #self.profiler.profile_operations(options=profile_op_opt_builder.build())
    
    #variables_names =[v.name for v in tf.trainable_variables() if v.name.startswith("prediction/l4")]
    #values = self.sess.run(variables_names)
    #for k,v in zip(variables_names, values):
    #    print(k, v)
    
    #variables_names =[v.name for v in tf.trainable_variables() if v.name.startswith("target/l4")]
    #values = self.sess.run(variables_names)
    #for k,v in zip(variables_names, values):
    #    print(k, v)

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def build_dqn_g(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.env.action_size, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=30, test_ep=0.05, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)
    scores = []
    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print("="*30)
      print(" [%d] current reward : %d" % (idx, current_reward))
      print(" [%d] Best reward : %d" % (best_idx, best_reward))
      print("="*30)
      scores.append(current_reward)
    print(scores)
    print(np.std(scores))
    print(np.mean(scores))
    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    #l4_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512, state_is_tuple=True)
    #l4_rnn_cell_g = tf.nn.rnn_cell.BasicLSTMCell(num_units=512, state_is_tuple=True)

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='s_t')
      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        #self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
        # self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
        #l5_in = tf.reshape(self.l4, [1] + tf.shape(self.l4)[0] + tf.shape(self.l4)[1])
        l4_in = tf.expand_dims(self.l3_flat, 0)
        #print(l5_in.get_shape())
        #l5_in = np.zeros([1] + self.l4.get_shape().as_list())
        #l5_in[0] = self.l4
        self.l4, self.w['l4_w'], self.w['l4_b'], l4_state = RNN(l4_in, 512, activation_fn=activation_fn, name='l4')
        self.l4 = self.l4[0]
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
        
        #final_state = state[0]
        #l4_guess_in = tf.concat(concat_dim=1,values=[final_state, action_plus_1_one_hot])
        self.action_sequence = tf.placeholder('int64', [None,None], name='action_sequence')
        action_sequence_one_hot = tf.one_hot(self.action_sequence, self.env.action_size, 1.0, 0.0, name='action_sequence_one_hot')
        self.l4_g, self.w['l4_g_w'], self.w['l4_g_b'], state = RNN(action_sequence_one_hot, 512, l4_state, activation_fn=activation_fn, name='l4_g')
        
        #predict_steps = [] # this will be the list of processed tensors
        #for each in tf.unstack(self.l4_g):
        #    # do whatever
        #    each_q_g, self.w['q_g_w'], self.w['q_g_b'] = linear(each, self.env.action_size, name='q_g')
        #    predict_steps.append(each_q_g)
        #self.q_g = tf.concat(predict_steps, 0)
        self.w['q_g_w'], self.w['l4_g_b']= define_linear(self.l4_g[0], self.env.action_size, name='q_g')
        self.q_g = tf.map_fn(lambda x : calculate_linear(x, self.env.action_size, name='q_g'), self.l4_g)
        #self.q_g = tf.expand_dims(self.q_g, 0)
        
      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')
      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        #self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            #linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        #self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            #linear(self.target_l4, self.env.action_size, name='target_q')
        #l5_in = tf.reshape(self.target_l4, [1] + tf.shape(self.target_l4)[0] + tf.shape(self.target_l4)[1])
        target_l4_in = tf.expand_dims(self.target_l3_flat, 0)
        #print(l5_in.get_shape())
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'], target_l4_state = RNN(target_l4_in, 512, activation_fn=activation_fn, name='l4')
        self.target_l4 = self.target_l4[0]
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = linear(self.target_l4, self.env.action_size, name='target_q')

        #final_state = state[0]
        #target_l4_guess_in = tf.concat(concat_dim=1,values=[final_state, target_action_plus_1_one_hot])
        #target_l4_guess_in = tf.expand_dims(target_action_plus_1_one_hot, 0)
        self.target_action_sequence = tf.placeholder('int64', [None,None], name='target_action_sequence')
        target_action_plus_one_hot = tf.one_hot(self.target_action_sequence, self.env.action_size, 1.0, 0.0, name='target_action_plus_one_hot')
        self.target_l4_g, self.t_w['l4_g_w'], self.t_w['l4_g_b'], state = RNN(target_action_plus_one_hot, 512, target_l4_state, activation_fn=activation_fn, name='l4_g')
        #self.target_l4_g = self.target_l4_g[0]
        #self.target_q_g, self.t_w['q_g_w'], self.t_w['q_g_b'] = linear(self.target_l4_g[0], self.env.action_size, name='target_q_g')
        #self.target_q_g = tf.expand_dims(self.target_q_g, 0)
        self.t_w['q_g_w'], self.t_w['l4_g_b']= define_linear(self.target_l4_g[0], self.env.action_size, name='target_q_g')
        self.target_q_g = tf.map_fn(lambda x : calculate_linear(x, self.env.action_size, name='target_q_g'), self.target_l4_g)

        
      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')
      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
      #step batch
      self.target_q_g_ts = tf.placeholder('float32', [None, None], name='target_q_g_ts')
      self.action_plus = tf.placeholder('int64', [None,None], name='action_plus')
      #self.action_plus_1 = tf.placeholder('int64', [None], name='action_plus_1')
      action_plus_one_hot = tf.one_hot(self.action_plus, self.env.action_size, 1.0, 0.0, name='action_plus_one_hot')
      #q_g_acted = tf.reduce_sum(self.q_g * action_one_hot, reduction_indices=1, name='q_g_acted')
      q_g_acted = tf.reduce_sum(self.q_g * action_plus_one_hot, reduction_indices=2, name='q_g_acted')

      self.delta = tf.subtract(self.target_q_t, q_acted)
      self.delta_g = tf.subtract(self.target_q_g_ts, q_g_acted)
      self.global_step = tf.Variable(0, trainable=False)

      self.loss_delta = tf.reduce_mean(clipped_error(self.delta), name='loss_delta')
      self.loss_delta_g = tf.reduce_mean(clipped_error(self.delta_g), name='loss_delta_g')
      self.loss = tf.add(tf.multiply(self.loss_delta, self.q_fraction), tf.multiply(self.loss_delta_g, self.q_guess_fraction), 'loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

