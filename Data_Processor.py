import numpy as np
class Data_Processor:
    def __init__(self,raw_data,apply_mean = True):
        self.train_len = self.padding_lens(raw_data,'train_data')
        self.test_len = self.padding_lens(raw_data,'test_data')
        self.val_len = self.padding_lens(raw_data,'val_data')

        train_acc = raw_data.datas['train_data']['acceleration']
        train_acc = np.array([np.array(data) for data in train_acc])
        train_ori = raw_data.datas['train_data']['orientation']
        train_ori = np.array([np.array(data) for data in train_ori])
        train_smpl = raw_data.datas['train_data']['smpl_pose']
        train_smpl = np.array([np.array(data) for data in train_smpl])

        test_acc = raw_data.datas['test_data']['acceleration']
        test_acc = np.array([np.array(data) for data in test_acc])
        test_ori = raw_data.datas['test_data']['orientation']
        test_ori = np.array([np.array(data) for data in test_ori])
        test_smpl = raw_data.datas['test_data']['smpl_pose']
        test_smpl = np.array([np.array(data) for data in test_smpl])

        val_acc = raw_data.datas['val_data']['acceleration']
        val_acc = np.array([np.array(data) for data in val_acc])
        val_ori = raw_data.datas['val_data']['orientation']
        val_ori = np.array([np.array(data) for data in val_ori])
        val_smpl = raw_data.datas['val_data']['smpl_pose']
        val_smpl = np.array([np.array(data) for data in val_smpl])

        self.mean_smpl = None
        self.mean_applied = False
        if apply_mean:
            self.mean_smpl = self.calculate_mean(train_smpl)
            train_smpl,test_smpl,val_smpl = self.apply_mean(train_smpl,test_smpl,val_smpl)

        self.train_data = np.array([np.hstack((
            train_acc[id],
            train_ori[id],
            train_smpl[id],
            self.train_len[id])) for id in range(self.train_len.shape[0]) ])
        self.test_data = np.array([np.hstack((
            test_acc[id],
            test_ori[id],
            test_smpl[id],
            self.test_len[id])) for id in range(self.test_len.shape[0]) ])
        self.val_data = np.array([np.hstack((
            val_acc[id],
            val_ori[id],
            val_smpl[id],
            self.val_len[id])) for id in range(self.val_len.shape[0]) ])
        


    def padding_lens(self,raw_data,data_set):
        first_len = raw_data.datas[data_set]['acceleration'][0].shape[0]
        pad = [0 for _ in range(first_len -1)]
        res_len = raw_data.datas[data_set]['seq_lens'].tolist()
        for id in range(len(res_len)):
            res_len[id] = np.array( [res_len[id]]+pad).reshape(-1,1)
        return np.array(res_len)

    def calculate_mean(self,data):
        return np.mean(data,axis = (1,0))

    def apply_mean(self,train_smpl,test_smpl,val_smpl):
        assert self.mean_smpl is not None
        assert self.mean_applied == False
        train_smpl -= self.mean_smpl
        test_smpl -= self.mean_smpl
        val_smpl -= self.mean_smpl
        self.mean_applied = True
        return train_smpl,test_smpl,val_smpl


    def help(self):
        print("##Data_Processor## members: train_len, test_len, val_len ## train_data, test_data, val_data")