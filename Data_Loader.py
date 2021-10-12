#initializing
import numpy as np
from numpy.lib.function_base import append
#paths
class IMU_npz:
    def __init__(self,data_root = './datas/'):
        self.data_root = data_root
        self.train_data = self.data_root + 'imu_own_training.npz'
        self.test_data = self.data_root + 'imu_own_test.npz'
        self.validation_data = self.data_root + 'imu_own_validation.npz'
        print("loading imu data in " + self.data_root)

#loader
class Data_Loader(object):
    def __init__(self,data_root = None, min_len = 100, equalize_clear_datalen = True, is_tolist = False):
        if data_root is None:
            self.path = IMU_npz()
        else:
            self.path = IMU_npz(data_root)
        assert self.path is not None, 'Try to input correct data path'
        self.min_len = min_len

        self.useful_attr = ['smpl_pose','acceleration','orientation','seq_lens']
        self.useful_sets = ['train_data','test_data','val_data']

        # tf.data.Dataset.from_tensor_slices

        train_data = np.load(self.path.train_data,allow_pickle = True)
        test_data = np.load(self.path.test_data,allow_pickle = True)
        val_data = np.load(self.path.validation_data,allow_pickle = True)

        self.datas = {
            'train_data':self.load_useful_attr(train_data),
            'test_data':self.load_useful_attr(test_data),
            'val_data':self.load_useful_attr(val_data)}
        
        self.append_len()
        
        if equalize_clear_datalen:
            self.padding_zero()
            self.check_equal_len_data()

        # tolist
        if is_tolist:
            for sets in self.useful_sets:
                cur_sets = self.datas[sets]
                for attr in self.useful_attr:
                    cur_sets[attr] = cur_sets[attr].tolist()
            
    
    def append_len(self):
        for key in self.useful_sets:
            cur_sets = self.datas[key]
            first_attr = cur_sets[self.useful_attr[0]]
            self.datas[key]['seq_lens'] = np.array([s.shape[0] for s in first_attr])

    def datas(self):
        return self.datas
    
    def load_useful_attr(self,data):
        res = {}
        for attr in ['smpl_pose','acceleration','orientation']:
            res[attr] = np.array([data for data in data[attr] if data.shape[0]>= self.min_len])
        return res
    
    def padding_zero(self):
        for key in self.useful_sets:
            res = self.datas[key]
            for attr in ['smpl_pose','acceleration','orientation']:
                maxlen = 0
                second_len = res[attr][0].shape[1]

                for seq in res[attr]:
                    assert seq.shape[1] == second_len
                    if seq.shape[0] > maxlen:
                        maxlen = seq.shape[0]

                for index,seq in enumerate(res[attr]):
                    dif = maxlen - seq.shape[0]
                    if dif > 0:
                        res[attr][index] = np.concatenate((res[attr][index],np.zeros((dif,second_len))))

    def check_equal_len_data(self,output = True):
        for key in self.useful_sets:
            cur_sets = self.datas[key]
            if output:
                print(key)
            for attr in ['smpl_pose','acceleration','orientation']:
                attrdata = cur_sets[attr]
                first_len = attrdata[0].shape[0]
                second_len = attrdata[0].shape[1]
                seq_len = attrdata.shape[0]
                if output:
                    print(attr + ' : ' + str(seq_len) + ' of ' + str([first_len,second_len]))
                for data in attrdata:
                    try:
                        if data.shape[0] != first_len or data.shape[1] != second_len:
                            print("wrong in sets : " + key)
                            break
                    except IndexError:
                        print('\n'+str(data))
        if output:
            print("All data checked, which lens are equal")