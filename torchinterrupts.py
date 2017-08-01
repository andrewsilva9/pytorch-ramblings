import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import os
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

batch_size = 1
epochs = 500
lr = 1e-5
log_interval = 1000
running_loss = []

use_gpu = torch.cuda.is_available()


def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


def save_checkpoint(state, filename='checkpoint_rel_small075.pth.tar'):
    torch.save(state, filename)


annots_dir = '/home/asilva/Data/int_annotations/my_rel_data/'
X = []
Y = []
bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
            'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
            'objects']
for filename in os.listdir(annots_dir):
    if os.path.isdir(os.path.join(annots_dir, filename)):
            continue
    af = open(os.path.join(annots_dir, filename), 'rb')
    annotation = pickle.load(af)
    # For each person in the segmentation
    for person in annotation:
        sorted_ann = sorted(annotation[person], key = lambda x: x['timestamp'])

        for piece in sorted_ann:
            new_input_data = []
            for key, value in piece.iteritems():
                if key in bad_keys:
                    continue
                if value >= 1.7e300:
                    piece[key] = -5
            label = piece['value']
            if label == 0:
                # last_value = -1
                continue
            new_input_data.append(piece['right_wrist_angle'])       # 0
            new_input_data.append(piece['right_elbow_angle'])       # 1
            new_input_data.append(piece['left_wrist_angle'])        # 2
            new_input_data.append(piece['left_elbow_angle'])        # 3
            # new_input_data.append(piece['shoulder_vec_x'])          # 4
            # new_input_data.append(piece['shoulder_vec_y'])          # 5
            new_input_data.append(piece['left_eye_angle'])          # 6
            new_input_data.append(piece['right_eye_angle'])         # 7
            # new_input_data.append(piece['eye_vec_x'])               # 8
            # new_input_data.append(piece['eye_vec_y'])               # 9
            new_input_data.append(piece['right_shoulder_angle'])    # 10
            new_input_data.append(piece['left_shoulder_angle'])     # 11
            new_input_data.append(piece['nose_vec_y'])              # 12
            new_input_data.append(piece['nose_vec_x'])              # 13
            # new_input_data.append(piece['left_arm_vec_x'])          # 14
            # new_input_data.append(piece['left_arm_vec_y'])          # 15
            # new_input_data.append(piece['right_arm_vec_x'])         # 16
            # new_input_data.append(piece['right_arm_vec_y'])         # 17
            new_input_data.append(piece['gaze'])                    # 18
            new_input_data.append(0)                                # book 19
            new_input_data.append(0)                                # bottle 20
            new_input_data.append(0)                                # bowl 21
            new_input_data.append(0)                                # cup 22
            new_input_data.append(0)                                # laptop 23
            new_input_data.append(0)                                # cell phone 24
            new_input_data.append(0)                                # blocks 25
            new_input_data.append(0)                                # tablet 26
            new_input_data.append(0)                                # unknown 27
            foi = 11 # first object index to make it easier to change stuff
            for item in piece['objects']:
                if item == 'book':
                    new_input_data[foi] += 1
                elif item == 'bottle':
                    new_input_data[foi+1] += 1
                elif item == 'bowl':
                    new_input_data[foi+2] += 1
                elif item == 'cup':
                    new_input_data[foi+3] += 1
                elif item == 'laptop':
                    new_input_data[foi+4] += 1
                elif item == 'cell phone':
                    new_input_data[foi+5] += 1
                elif item == 'blocks':
                    new_input_data[foi+6] += 1
                elif item == 'tablet':
                    new_input_data[foi+7] += 1
                else:
                    new_input_data[foi+8] += 1
            # if sum(new_input_data) < 0:
            #     continue
            if label <= 2:
                label = 0
            else:
                label = 1
            X.append(new_input_data)
            Y.append(label)


# annots_dir = '/Users/andrewsilva/my_annots_trimmed'
# X = []
# Y = []
# bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
#             'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
#             'objects']
# for filename in os.listdir(annots_dir):
#     af = open(os.path.join(annots_dir, filename), 'rb')
#     annotation = pickle.load(af)
#     # For each person in the segmentation
#     for person in annotation:
#         sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
#         for piece in sorted_ann:
#             new_input_data = []
#             for key, value in piece.iteritems():
#                 if key in bad_keys:
#                     continue
#                 if value >= 1.7e300:
#                     piece[key] = -1
#             label = piece['value']
#             if label == 0:
#                 last_value = -1
#                 continue
#             new_input_data.append(piece['left_shoulder_x'])
#             new_input_data.append(piece['left_shoulder_y'])
#             new_input_data.append(piece['left_elbow_x'])
#             new_input_data.append(piece['left_elbow_y'])
#             new_input_data.append(piece['left_wrist_x'])
#             new_input_data.append(piece['left_wrist_y'])
#             new_input_data.append(piece['left_eye_x'])
#             new_input_data.append(piece['left_eye_y'])
#             new_input_data.append(piece['right_shoulder_x'])
#             new_input_data.append(piece['right_shoulder_y'])
#             new_input_data.append(piece['right_elbow_x'])
#             new_input_data.append(piece['right_elbow_y'])
#             new_input_data.append(piece['right_wrist_x'])
#             new_input_data.append(piece['right_wrist_y'])
#             new_input_data.append(piece['right_eye_x'])
#             new_input_data.append(piece['right_eye_y'])
#             new_input_data.append(piece['nose_x'])
#             new_input_data.append(piece['nose_y'])
#             new_input_data.append(piece['neck_x'])
#             new_input_data.append(piece['neck_y'])
#             new_input_data.append(piece['gaze'])
#             new_input_data.append(0)  # book 21
#             new_input_data.append(0)  # bottle 22
#             new_input_data.append(0)  # bowl 23
#             new_input_data.append(0)  # cup 24
#             new_input_data.append(0)  # laptop 25
#             new_input_data.append(0)  # cell phone 26
#             new_input_data.append(0)  # blocks 27
#             new_input_data.append(0)  # tablet 28
#             new_input_data.append(0)  # unknown 29
#             for item in piece['objects']:
#                 if item == 'book':
#                     new_input_data[21] += 1
#                 elif item == 'bottle':
#                     new_input_data[22] += 1
#                 elif item == 'bowl':
#                     new_input_data[23] += 1
#                 elif item == 'cup':
#                     new_input_data[24] += 1
#                 elif item == 'laptop':
#                     new_input_data[25] += 1
#                 elif item == 'cell phone':
#                     new_input_data[26] += 1
#                 elif item == 'blocks':
#                     new_input_data[27] += 1
#                 elif item == 'tablet':
#                     new_input_data[28] += 1
#                 else:
#                     new_input_data[29] += 1
#             if sum(new_input_data) < 0:
#                 continue
#             if label <= 2:
#                 label = [1, 0]
#             else:
#                 label = [0, 1]
#             X.append(torch.FloatTensor(new_input_data))
#             Y.append(torch.FloatTensor(label))
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.relu1 = nn.ReLU()
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        # self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(100, 50)
        self.relu3 = nn.ReLU()
        # self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(50, 10)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(10, 2)
        self.soft = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 20)
        x = self.soft(self.fc5(self.relu4(self.fc4(self.relu3(self.fc3(self.relu2(self.fc2(self.relu1((self.fc1(x)))))))))))
        # x = self.soft(self.fc5(self.relu2(self.fc2(self.relu1(self.fc1(x))))))
        return x
if use_gpu:
	model = Net().cuda()
else:
	model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if use_gpu:
	loss_fn = torch.nn.NLLLoss(weight=torch.cuda.FloatTensor([1, 5]), size_average=False)
else:
	loss_fn = torch.nn.NLLLoss(weight=torch.FloatTensor([1, 5]), size_average=False)

# loss_fn = torch.nn.CrossEntropyLoss(size_average=False)


def train(epoch):
    model.train()
    avg_loss = 0
    count = 0
    for iteration, batch in enumerate(train_loader, 1):
    	if use_gpu:
        	data, target = Variable(batch[0].cuda()), Variable(batch[1].cuda())
        else:
        	data, target = Variable(batch[0]), Variable(batch[1])

        # output = model(data)
        loss = loss_fn(model(data), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % log_interval == 0:
            print 'Train Epoch', epoch, ' || Batch Index:', iteration, ' || Loss:', loss.data[0]
        avg_loss += loss.data[0]
        count += 1.
    return avg_loss/count


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    true_zero = 0
    true_one = 0
    false_zero = 0
    false_one = 0
    for iteration, batch in enumerate(test_loader, 1):
    	if use_gpu:
        	data, target = Variable(batch[0].cuda()), Variable(batch[1].cuda())
        else:
        	data, target = Variable(batch[0]), Variable(batch[1])
        output = model(data)
        loss = loss_fn(output, target)
        test_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for i in range(int(output.size()[0])):
            pred = output.data.max(1)[1][i]
            correct += pred.eq(target.data[i]).cpu().sum()
            if pred[0] == 0 and target.data[i] == 0:
                true_zero += 1.
            elif pred[0] == 0 and target.data[i] == 1:
                false_zero += 1.
            elif pred[0] == 1 and target.data[i] == 1:
                true_one += 1.
            elif pred[0] == 1 and target.data[i] == 0:
                false_one += 1.

    test_loss /= len(test_loader.dataset)
    print 'NoScaler NoDrop NLL5 SA = F: Average test loss:', test_loss, ' || Accuracy:', (100.*correct/len(test_loader.dataset))

    total_zero = true_zero + false_one
    total_one = true_one + false_zero
    print true_zero/total_zero, ' ', false_one/total_zero
    print false_zero/total_one, ' ', true_one/total_one
    return test_loss

# split_point = int(len(X) * 0.8)
# X_train, X_test = X[:split_point], X[split_point:]
# y_train, y_test = Y[:split_point], Y[split_point:]
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# for index in range(len(X_train)):
#     real_x_train.append(X_train[index])
# for index in range(len(X_test)):
#     real_x_test.append(X_test[index])
kf = StratifiedKFold(n_splits=5, shuffle=False)

fold_number = 1
for train_idx, test_idx in kf.split(X, Y):
    real_x_train = []
    real_x_test = []
    y_train = []
    y_test = []
    for idx in train_idx:
        real_x_train.append(X[idx])
        y_train.append(Y[idx])
    for idx in test_idx:
        real_x_test.append(X[idx])
        y_test.append(Y[idx])
    if use_gpu:
    	real_x_train = torch.cuda.FloatTensor(real_x_train)
    	y_train = torch.cuda.LongTensor(y_train)
    	real_x_test = torch.cuda.FloatTensor(real_x_test)
    	y_test = torch.cuda.LongTensor(y_test)
    else:
    	real_x_train = torch.Tensor(real_x_train)
    	y_train = torch.LongTensor(y_train)
    	real_x_test = torch.FloatTensor(real_x_test)
    	y_test = torch.LongTensor(y_test)

    train_dat = data_utils.TensorDataset(real_x_train, y_train)
    train_loader = data_utils.DataLoader(train_dat, batch_size=2, shuffle=False)

    
    test_dat = data_utils.TensorDataset(real_x_test, y_test)
    test_loader = data_utils.DataLoader(test_dat, batch_size=2, shuffle=False)

    train_loss = []
    test_loss = []
    for epoch in range(epochs):
    	if epoch % 50 == 0:
        	print 'Fold#: ', fold_number
        train_loss.append(train(epoch))
        test_loss.append(test(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
    fold_number += 1
    # plt.plot(train_loss, 'r', test_loss, 'b')
    # plt.show()
