from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        # TODO: 在Windows系统下，由于多进程序列化的限制，PyTorch的DataLoader不能直接使用lambda函数作为collate_fn
        #   要么num_workers设为0，要么把lambda函数改写成标准python函数格式
        # Windows / macOS使用spawn方式创建子进程，需要通过pickle序列化所有传递给子进程的对象
        # 局部函数（定义在其他函数内部的函数）无法被pickle，因为它们依赖于外部函数的局部命名空间
        # 全局函数和类实例（只要它们的属性和方法也可序列化）可以被pickle
        # 要改写成以下方式
        # class CustomCollate:
        #     def __init__(self, max_len):
        #         self.max_len = max_len
        #
        #     def __call__(self, x):
        #         return collate_fn(x, max_len=self.max_len)
        #
        # def data_provider(args):
        #     # 初始化 collate_fn 实例
        #     collate_fn_instance = CustomCollate(max_len=args.seq_len)
        #
        #     # 使用实例作为 collate_fn
        #     data_loader = DataLoader(
        #         data_set,
        #         batch_size=args.batch_size,
        #         shuffle=True,
        #         num_workers=args.num_workers,
        #         collate_fn=collate_fn_instance
        #     )
        #     return data_loader

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            # shuffle=False,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            # collate_fn=custom_collate,
        )
        # 此collate_fn将把所有实例在时序长度上进行统一，按照所有实例的最长长度max_len
        # 具体做法就是填充0，比如第1个数据时序长度只有19，最大长度为29，则往后填充10个0，
        # 并且随之生成一个二值mask，mask为True代表真实值，False代表填充值，前19为True，后10为False
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
