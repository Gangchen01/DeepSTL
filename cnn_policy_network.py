import tensorflow as tf
import numpy as np
import random
import argparse
import os
import multiprocessing
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('T', 20,
"""x <=T.""")
tf.app.flags.DEFINE_integer('UP', 20,
"""Z<=UP.""")

tf.app.flags.DEFINE_integer('LB', 0,
"""Z >=LB.""")


class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_size, signal_length):
        with tf.variable_scope(name):
            self._init(ob_size, signal_length)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_size,signal_length):

        self._graph = tf.get_default_graph()
        with self._graph.as_default():
            self._ob =tf.placeholder(name="ob", dtype=tf.float32, shape=[None] + list([ob_size]))

            x = self._ob
            x = tf.layers.dense(inputs=x,
                                     units=256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.nn.l2_loss)
            x = tf.layers.dense(inputs=x,
                                     units=256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.nn.l2_loss)
            x = tf.layers.dense(inputs=x,
                                     units=256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.nn.l2_loss)

            # the output size
            # num of first size
            self._disc_logits0= tf.layers.dense(x, 2, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            #Num of second size
            self._disc_logits1= tf.layers.dense(x, 2, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            #Num of third size
            self._disc_logits2= tf.layers.dense(x, signal_length, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            #Num of forth size
            self._disc_logits3= tf.layers.dense(x, 4, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

            #Num of fivth size
            self._disc_logits4= tf.layers.dense(x, 3, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # continous size
            continous_num=3
            self._cons_logits= tf.layers.dense(x, continous_num*2, name='logits1',activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # the state-value function
            self._value_pred=tf.layers.dense(x, 1,activation=tf.nn.relu, name='state_value', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.state_in = []
            stochastic = tf.placeholder(dtype=tf.bool, shape=())





            self._sess=self._set_sess()

        return
    def _set_sess(self,graph=None):
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
        return tf.Session(config=tf_config, graph=self._graph)

    def neglogp_con(self,x,mean_logstd):
        #print(mean_logstd)
        mean_logstd=tf.reshape(mean_logstd,shape=[-1,3,2])
        mean,logstd=tf.split(mean_logstd, num_or_size_splits=2,axis=-1)

        std=tf.exp(logstd)
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(logstd, axis=-1)
    def neglogp_dis(self,x,s0,s1,s2,s3,s4):

        x0,x1,x2,x3,x4=tf.split(-1,5,x)

        p0=tf.gather(s0,x0,axis=-1)
        #print("shape of p0",s0)
        p1=tf.gather(s1,x1,axis=-1)
        p2=tf.gather(s2,x2,axis=-1)
        p3=tf.gather(s3,x3,axis=-1)
        p4=tf.gather(s4,x3,axis=-1)
        p_dis=p0*p1*p2*p3*p4
        neglogp_dis_val=-tf.log(p_dis)

        return  neglogp_dis_val
    def get_logp(self,ac):
        ac_dis=ac[...,:5]
        ac_dis=tf.cast(ac_dis, tf.int32)  # [1, 2], dtype=tf.int32

        ac_con=ac[...,-6:]
        softmax0=tf.nn.softmax(self._disc_logits0)
        softmax1=tf.nn.softmax(self._disc_logits1)
        softmax2=tf.nn.softmax(self._disc_logits2)
        softmax3=tf.nn.softmax(self._disc_logits3)
        softmax4=tf.nn.softmax(self._disc_logits4)

        neg_logp_dis_val=self.neglogp_dis(ac_dis,softmax0,softmax1,softmax2,softmax3,softmax4)
        neg_logp_con_val=self.neglogp_con(ac_con,self._cons_logits)

        neg_logp=neg_logp_dis_val+neg_logp_dis_val
        return neg_logp
    def get_action_reward(self, state_val,sess):
        softmax0=tf.nn.softmax(self._disc_logits0)
        softmax1=tf.nn.softmax(self._disc_logits1)
        softmax2=tf.nn.softmax(self._disc_logits2)
        softmax3=tf.nn.softmax(self._disc_logits3)
        softmax4=tf.nn.softmax(self._disc_logits4)
        feed_dict={self._ob: state_val}
        output=(softmax0,softmax1,softmax2,softmax3,softmax4,self._cons_logits,self._value_pred)
        soft0_val,soft1_val,soft2_val,soft3_val,soft4_val,cons_logits_val,pre_val=sess.run(output,feed_dict=feed_dict)

        act_dis=self.rand_chose_dis(soft0_val,soft1_val,soft2_val,soft3_val,soft4_val)
        act_cons=self.rand_chose_con(cons_logits_val)
        print(act_dis.shape)
        print(act_cons.shape)

        actions_combine=np.concatenate((act_dis,act_cons),axis=1)

        return actions_combine, pre_val
    def rand_chose_dis(self,s0,s1,s2,s3,s4):
        shape_dis=s0.shape

        if len(shape_dis)==2:
            act_dis = np.zeros((shape_dis[0],5))

            for i in range(shape_dis[0]):
                    #print(s0[i])
                    #print(np.random.choice(list(range(s1.shape[-1])),size= 1, p=s1[i]))

                    act_dis[i, 0] = np.random.choice(list(range(s0.shape[-1])),size= 1, p=s0[i])

                    act_dis[i, 1] = np.random.choice(list(range(s1.shape[-1])) ,size= 1, p=s1[i])
                    act_dis[i, 2] = np.random.choice(list(range(s2.shape[-1])), size= 1, p= s2[i])
                    act_dis[i, 3] = np.random.choice(list(range(s3.shape[-1])), size= 1, p= s3[i])
                    act_dis[i, 4] = np.random.choice(list(range(s4.shape[-1])), size= 1, p= s4[i])
        else:
            act_dis = np.zeros((5))

            act_dis[ 0] = np.random.choice(list(range(s0.shape[-1])), 1, s0)
            act_dis[ 1] = np.random.choice(list(range(s1.shape[-1])), 1, s1)
            act_dis[ 2] = np.random.choice(list(range(s2.shape[-1])), 1, s2)
            act_dis[ 3] = np.random.choice(list(range(s3.shape[-1])), 1, s3)
            act_dis[ 4] = np.random.choice(list(range(s4.shape[-1])), 1, s4)
        return act_dis
    def get_vpred(self):
        return self._value_pred
    def rand_chose_con(self,con0):
        con0=np.reshape(con0,[-1,3,2])
        shape_con=con0.shape
        if len(shape_con)==3:
            act_con = np.zeros(shape_con[0:2])
            #print(act_con.shape)
            for i in range(shape_con[0]):
                for j in range(shape_con[1]):
                    #print(con0[i,j])
                    act_con[i,j]=self.gauss_sample(con0[i,j])

        else:
            act_con = np.zeros(shape_con)
            for i in range(shape_con[0]):
                act_con[i] = self.gauss_sample(con0[i])

        return act_con
    def gauss_sample(self,mean_std):

        sample_result=mean_std[0]+np.exp(mean_std[1])*np.random.normal()
        return sample_result

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class  Get_Datas(object):

    def __init__(self,pi,ob_size, trajt_num,depth,batch_size, sess, gamba=0.9):
        self._pi=pi
        self._ob_size=ob_size
        self._trajt_num=trajt_num
        self._depth=depth
        self._gamba=gamba
        self._sess=sess
        self._batch_size=batch_size
        self.update(pi)


        #atarg_data = (atarg_data - atarg_data.mean()) / atarg_data.std()

    def get_data(self,batch_size):
        lenth=self._depth*self._trajt_num
        index=np.random.randint(0,lenth,batch_size)
        ob_data=[]
        ac_data=[]
        atarg_data=[]
        ret_data=[]
        for i in range(len(index)):
            ob_data.append(np.take(self._stat_set,index[i]))
            ac_data.append(np.take(self._ac_set,index[i]))
            atarg_data.append(np.take(self._adv_set,index[i]))
            ret_data.append(np.take(self._returns_set,index[i]))
        ob_data=np.array(ob_data)
        ac_data=np.array(ac_data)
        atarg_data=np.array(atarg_data)
        ret_data=np.array(ret_data)
        return ob_data,ac_data,atarg_data,ret_data
    def get_stat(self,actions):
        shape_ac=actions.shape
        act_with_model=np.zeros((shape_ac[0],11))
        for i in range(shape_ac[0]):

                act_with_model[i,0]=actions[i,0]+3
                act_with_model[i, 1] = 0
                act_with_model[i, 2] = actions[i,1]*5
                act_with_model[i, 3:5] = [0,0]
                act_with_model[i, 7] = actions[i,2]
                act_with_model[i, 8] = actions[i,3]
                act_with_model[i, 10] = actions[i,4]+1

                act_with_model[i,5]=min(FLAGS.T, actions[i,5])
                act_with_model[i, 6] = min(FLAGS.T, actions[i,5]+actions[i, 6])
                act_with_model[i, 9] = min(FLAGS.UP, actions[i, 7] )
                act_with_model[i, 9] = max(FLAGS.LB, act_with_model[i, 9])

        stat_new= model_get_stat(act_with_model)################============
        return stat_new
    def get_returns(self,stat):
        #returns=#get return from model
        size=len(returns)
        returns_sets=np.zeros((size,self._depth))
        for i in reversed(range(self._depth)):
            returns_sets[i]=returns_sets*gamba**(self._depth-i-1)
        return returns_sets
    def update(self,pi):
        self._ac_set=[]
        self._reward_set=[]
        init_stat=np.zeros((self._trajt_num,self._ob_size))
        self._ac_zeros,self._predv_zeros=self._pi.get_action_reward(init_stat,self._sess)
        self._ac_set=[]

        self._ac_set.append(self._ac_zeros)
        self._reward_set.append(self._predv_zeros)
        self._stat_set=[]

        stat=self.get_stat(self._ac_zeros)
        self._stat_set.append(stat)

        for i in range(self._depth):
            actions,reward=self._pi.get_action_reward(stat,self._sess)
            stat = self.get_stat(actions)
            self._reward_set.append(reward)
            self._ac_set.append(stat)
            self._stat_set.append(actions)

        self._returns_set=self.get_returns(stat)
        self._ac_set=np.array(self._ac_set)
        self._stat_set=np.array(self._stat_set)
        self._reward_set=np.array(self._reward_set)
        self._adv_set=self._returns_set-self._reward_set
        return





def get_placeholder_cached(name,ob_size):
    _PLACEHOLDER_CACHE = {}

    name1 = "ob"
    dtype = tf.float32
    shape =[None,ob_size]
    out = tf.placeholder(dtype=dtype, shape=shape, name=name)

    _PLACEHOLDER_CACHE[name1] = (out, dtype, shape)

    return _PLACEHOLDER_CACHE[name][0]

def action_placeholder():

    dis=5
    cons=6
    return tf.placeholder(tf.float32,[None,dis+cons])
def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])
def assign_old_eq_new(pi,oldpi):
    newv=pi.get_variables()
    oldv=oldpi.get_variables()


    tf.assign(oldv, newv)
    return pi,oldpi

def net_learn():

    depth=4
    ob_size=depth*6+1
    signal_length=100
    batch_size=128
    trajt_num=50
    depth_num=4
    max_step=1e+5
    maxvpred=0
    maxvpred_stat=0
    optim_epochs=4
    pi=CnnPolicy("pi",ob_size=ob_size,signal_length=signal_length)
    oldpi=CnnPolicy("piold",ob_size=ob_size,signal_length=signal_length)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    clip_param=0.2
    ob=get_placeholder_cached("ob",ob_size)
    ac=action_placeholder()
    ratio = tf.exp(pi.get_logp(ac) - oldpi.get_logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.get_vpred() - ret))
    total_loss = pol_surr  + vf_loss
    var_list = pi.get_trainable_variables()
    vpred=pi.get_vpred()
    #flatgrad_value=flatgrad(total_loss,var_list)
    lr=1e-3
    opt=tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-05,
        use_locking=False,
        name='Adam'
        )
    varandgrad=opt.compute_gradients(total_loss,var_list=var_list)
    learning_opt=opt.apply_gradients(varandgrad)
    timesteps_so_far=0
    init_var=tf.global_variables()
    init_op=tf.variables_initializer(init_var)


    num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    with tf.Session(config=tf_config) as sess:
        sess.run(init_op)

        data_object = Get_Datas(pi, ob_size, trajt_num=trajt_num, depth=depth_num, batch_size=batch_size,gamba=0.9,sess=sess)
        saver=tf.train.Saver(max_to_keep=40)
        pi, oldpi = assign_old_eq_new(pi, oldpi)  # before learn

        while True:
            print("the max timestep is :",timesteps_so_far)
            if timesteps_so_far>max_step:
                break
            #get train value
            #ob_data,ac_data,atarg_data,ret_data=data_object.get_datas(batch_size)
            #learning for eptim_epochs
            for i in range(optim_epochs):
                ob_data, ac_data, atarg_data, ret_data= data_object.get_datas(batch_size)#ob_data, ac_data, atarg_data, ret_data
                feed_dict={ob:ob_data, ac:ac_data, atarg:atarg_data, ret: ret_data}

                _,summary_str=sess.run((learning_opt,summary_op),feed_dict=feed_dict)
            #evaluate
            output=(vpred,total_loss)
            ob_data, ac_data, atarg_data, ret_data = data_object.get_datas(
                batch_size)  # ob_data, ac_data, atarg_data, ret_data
            feed_dict = {ob: ob_data, ac: ac_data, atarg: atarg_data, ret: ret_data}

            vpred_val,loss_val=sess.run(output,feed_dict=feed_dict)
            max_index=np.nanargmax(vpred_val)
            if vpred_val.max()>maxvpred:
                maxvpred=vpred_val.max()
                print("the max vpred is :",maxvpred)
                maxvpred_stat=ob_data[max_index]
                print("the max vpred_stat is :",maxvpred_stat)

            pi, oldpi = assign_old_eq_new(pi, oldpi)  # after learn
            timesteps_so_far = timesteps_so_far + 1
            data_object.update(oldpi)
            if timesteps_so_far%200==0:
                checkpoint_file=("net_saved/model.ckpt")
                saver.save(sess, checkpoint_file, global_step=timesteps_so_far)


        #save file




if __name__=="__main__":
    net_learn()