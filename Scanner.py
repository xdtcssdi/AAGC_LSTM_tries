import random
class Scanner:
    def __init__(self,train_len = 1,future_len=40,past_len = 60):
        '''
        train_len: means how many frames are to be predict
        futrue_len: number of futrue frames that is to be put into the LSTM
        past_len: number of past frames that is to be put into the LSTM
        '''
        assert train_len >= 1,"train_len must larger than one"
        self.train_len = train_len
        self.future_len = future_len
        self.past_len = past_len
        self.len_of_frames = self.train_len + self.future_len + self.past_len -1

    def random_fetch_arraylike(self,data_seq):
        data_len = int(data_seq[0][-1])
        random_mask = data_len - self.len_of_frames
        if random_mask < 0:
            return None
        start = random.randint(0,random_mask)
        return data_seq[start:start + self.len_of_frames]

    def linear_scan(self,data_seq,data_lens):
        # TODO: maybe bug in this block, not tunned yet
        res = []
        for id,len in enumerate(data_lens):
            mask = len - self.len_of_frames
            if mask <0:
                continue
            res += [data_seq[id][start:start + self.len_of_frames] for start in range(mask-1)]
        return res