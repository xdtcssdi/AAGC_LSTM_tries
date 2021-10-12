import numpy as np
import tensorflow as tf
class IMU_Dataset:
    def __init__(self,processed_data,scanner,repeat = 10,shuffle_buffer = 10000,num_batch = 32):
        self.datas = processed_data
        self.repeat = repeat
        self.shuffle_buffer = shuffle_buffer
        self.num_batch = num_batch

        self.reset_train()
        self.reset_test()
        self.reset_val()

        self.scanner = scanner

    def reset_train(self):
        self.train_set = tf.data.Dataset.from_tensor_slices(self.datas.train_data
            ).repeat(self.repeat).shuffle(self.shuffle_buffer).batch(self.num_batch).as_numpy_iterator()
    def reset_test(self):
        self.test_set = tf.data.Dataset.from_tensor_slices(self.datas.test_data
            ).repeat(self.repeat).shuffle(self.shuffle_buffer).batch(self.num_batch).as_numpy_iterator()
    def reset_val(self):
        self.val_set = tf.data.Dataset.from_tensor_slices(self.datas.val_data
            ).repeat(self.repeat).shuffle(self.shuffle_buffer).batch(self.num_batch).as_numpy_iterator()

    def train_batch(self):
        data = self.train_set.next()
        res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        # while res is None:
        #     data = self.train_set.next()
        #     res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        return res[:,:,:-1]
    
    def test_batch(self):
        data = self.train_set.next()
        res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        # while res is None:
        #     data = self.train_set.next()
        #     res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        return res[:,:,:-1]
    
    def val_batch(self):
        data = self.train_set.next()
        res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        # while res is None:
        #     data = self.train_set.next()
        #     res = np.array(list(map(self.scanner.random_fetch_arraylike, data)))
        return res[:,:,:-1]

    def help(self):
        print("##IMU_Dataset## reset dataset by : reset_xx() method, get batch data by : xx_batch() method")