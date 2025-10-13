from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class MAPPODataset(Dataset):
    def __init__(self, co_data) -> None:
        super().__init__()
        (self.train_x, self.train_x_cat, self.train_y, self.train_hour, self.train_week, 
         self.train_userid, self.train_hour_pre, self.train_week_pre, self.train_timestamp, 
         self.train_geo, self.user_long_term_hash, self.poi_long_term_hash) = co_data

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, index):
        return self.get_feed_dict(index)

    def get_feed_dict(self, index):
        feed_dict = {
            'x': self.train_x[index],
            'x_cat': self.train_x_cat[index],
            'y': self.train_y[index],
            'hour': self.train_hour[index],
            'week': self.train_week[index],
            'userid': self.train_userid[index],
            'hour_pre': self.train_hour_pre[index],
            'week_pre': self.train_week_pre[index],
            'timestamp': self.train_timestamp[index],
            'geo': self.train_geo[index],
            'user_long_term_hash': self.user_long_term_hash[index],
            'poi_long_term_hash': self.poi_long_term_hash[index]
        }
        return feed_dict


def custom_collate_fn(batch):

    # 分离所有字段并转换为张量
    x_list = [torch.as_tensor(item['x']) for item in batch]
    x_cat_list = [torch.as_tensor(item['x_cat']) for item in batch]
    y_list = [torch.as_tensor(item['y']) for item in batch]
    hour_list = [torch.as_tensor(item['hour']) for item in batch]
    week_list = [torch.as_tensor(item['week']) for item in batch]
    userid_list = [torch.as_tensor(item['userid']) for item in batch]
    hour_pre_list = [torch.as_tensor(item['hour_pre']) for item in batch]
    week_pre_list = [torch.as_tensor(item['week_pre']) for item in batch]
    timestamp_list = [torch.as_tensor(item['timestamp']) for item in batch]
    geo_list = [torch.as_tensor(item['geo']) for item in batch]
    user_long_term_hash_list = [torch.as_tensor(item['user_long_term_hash']) for item in batch]
    poi_long_term_hash_list = [torch.as_tensor(item['poi_long_term_hash']) for item in batch]
    
    # 对于变长序列，使用pad_sequence进行填充
    x_padded = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True, padding_value=0)
    x_cat_padded = torch.nn.utils.rnn.pad_sequence(x_cat_list, batch_first=True, padding_value=0)
    hour_padded = torch.nn.utils.rnn.pad_sequence(hour_list, batch_first=True, padding_value=0)
    week_padded = torch.nn.utils.rnn.pad_sequence(week_list, batch_first=True, padding_value=0)
    hour_pre_padded = torch.nn.utils.rnn.pad_sequence(hour_pre_list, batch_first=True, padding_value=0)
    week_pre_padded = torch.nn.utils.rnn.pad_sequence(week_pre_list, batch_first=True, padding_value=0)
    
    # 对于固定长度的张量，直接stack
    y = torch.stack(y_list)
    userid = torch.stack(userid_list)
    timestamp = torch.stack(timestamp_list)
    geo = torch.stack(geo_list)
    user_long_term_hash = torch.stack(user_long_term_hash_list)
    poi_long_term_hash = torch.stack(poi_long_term_hash_list)
    
    return {
        'x': x_padded,
        'x_cat': x_cat_padded,
        'y': y,
        'hour': hour_padded,
        'week': week_padded,
        'userid': userid,
        'hour_pre': hour_pre_padded,
        'week_pre': week_pre_padded,
        'timestamp': timestamp,
        'geo': geo,
        'user_long_term_hash': user_long_term_hash,
        'poi_long_term_hash': poi_long_term_hash
    }
