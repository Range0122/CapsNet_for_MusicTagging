PureCapsNet_0.8241
Test scores: rocauc=0.824106	prauc=0.300251	acc=0.052110	f1=0.313359
输入大小为（96，96），因为训练速度太慢了，一轮大概1.4h，但是可以堆时间试试看有没有突破，目前感觉可能性不大，接近30epoch下来的变化如下：
(epoch=10, roc-auc=0.779705)
(epoch+=7, roc-auc=0.803679)
(epoch+=9, roc-auc=0.817352）
(epoch+=9, roc-auc=0.824106)
实验参数：
CNN（128， 128） 9x9 stride=3，2 
dim_capsule = 10
n_channels = 16


PureCapsNet_0.8010.h5
Test scores: rocauc=0.801036	prauc=0.257739	acc=0.040081	f1=0.228670 已经训练了10轮，每轮4h
输入（96，96），CNN为（128，128，128），5x5的kernel，stride=2
n_channels=32， dim_capsule=16


PureCapsNet_0.7996.h5
Test scores: rocauc=0.799572	prauc=0.259323	acc=0.031796	f1=0.220415
(96,96), (128,128,128) 3x3 2 
n_channels=16， dim_capsule=10
【routings=2，感觉效果不大，所以先停了，猜测跟0.8241区别不大】
(epoch=10, roc-auc=0.7996)

PureCapsNet_0.7866.h5
Test scores: rocauc=0.786578	prauc=0.236307	acc=0.029488	f1=0.155552
(96,96), (128,128,128) 3x3 2 
n_channels=16， dim_capsule=10
【routings=1】
(epoch=6, roc-auc=0.7866)


MixCapsNet 训练了8轮 有点不增长了   用的marginloss
Test scores: rocauc=0.833206	prauc=0.316474	acc=0.056069	f1=0.327459

下面换成binary_crossentropy
Test scores: rocauc=0.841771	prauc=0.318304	acc=0.057531	f1=0.202625

=========================================
Test scores: rocauc=0.878898	prauc=0.370301	acc=0.077793	f1=0.292674

MixCapsNet_0.890832.h5  Adam
然后转换为sgd继续训练 得到0.8987
具体的可以在self-attention那篇论文里面看到

上面的记不清楚了，下面是Basic_CNN的效果
Test scores: rocauc=0.889942	prauc=0.405491	acc=0.086796	f1=0.373694


