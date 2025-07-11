1、环境方面：
    Linux系统（ubuntu20.04） + CUDA10.2 + Cudnn7.6.5
    pytorch1.12.1（用到了torch.optim.RAdam这个比较新的优化器，1.8以前都不支持，索性安装一个比较新的版本）
    其他就按requirements里来
    注意，你的cudatoolkit如果跟CUDA版本不一致，那你安装pytorch的时候就会默认给你安装cpu版本，需要先检查cudatoolkit版本与nvcc是否一致

    如果是windows系统，torch的自动混合精度计算模块amp是在torch.cuda.amp中，而linux版的torch在torch.amp中，要到三方库源码中修改位置。
    并且windows系统的pillow不会根据你改变pytorch版本而自动更新，需要手动升级或者降级pillow的版本

2、数据集下载之后，放到根目录下面的dataset文件夹中（没有就创建一个，在scripts里有每个任务的启动sh，里面写了数据集路径）