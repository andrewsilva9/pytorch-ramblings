import torch
from torch.autograd import Variable
import os
import pickle

annots_dir = '/Users/andrewsilva/my_rel_data/'

X = []
Y = []
bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
            'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
            'objects']
for filename in os.listdir(annots_dir):
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
                    piece[key] = -1
            label = piece['value']
            if label == 0:
                last_value = -1
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

            if label <= 2:
                label = [1, 0]
            else:
                label = [0, 1]
            X.append(torch.FloatTensor(new_input_data))
            Y.append(torch.FloatTensor(label))


split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = Y[:split_point], Y[split_point:]
X_test = Variable(torch.FloatTensor(X_test))
X_train = Variable(torch.FloatTensor(X_train))
y_train = Variable(torch.FloatTensor(y_train))
y_test = Variable(torch.FloatTensor(y_test))
batch_size = 1
dim_in = 20
dim_hid = 100
dim_out = 2

model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_hid),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_hid, dim_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for it in range(500):
    y_pred = model(X)

    loss = loss_fn(y_pred, Y)
    print(it, loss.data[0])

    optimizer.zero_grad() # Let optimizer solve
    # model.zero_grad() # Manually update params

    loss.backward()

    optimizer.step() # Let optimizer solve
    # for param in model.parameters(): # Manually update params
    #     param.data -= lr * param.grad.data