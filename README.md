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
