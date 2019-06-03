from training.neural_network_adaptor import NN_training_adaptor

nadpt = NN_training_adaptor()
nadpt.load_data( '../data/train_x.npy', '../data/train_y.npy', '../data/test_x.npy', '../data/test_y.npy')
comb = [(1,0),(1,0),(1,1),(1,2),(1,3),(1,4)]
path = ['../models/NN00','../models/NN10','../models/NN11','../models/NN12','../models/NN13','../models/NN14']
for x,y in zip(comb,path):
    x_train, y_train, label_num = nadpt.label_data(layer=x[0], expand=x[1],X_data=nadpt.X_train, Y_data=nadpt.Y_train)
    x_train, y_train, x_val, y_val = nadpt.create_dataset(X_train= x_train, Y_train=y_train, num_labels=label_num)
    nadpt.network(cls_num=label_num)
    nadpt.train_model(X_train=x_train,Y_train=y_train,x_val=x_val,y_val=y_val,save_path=y)
    # nadpt.load_model('../models/test')
    # x_test,y_test,_ = nadpt.label_data(layer=x[0], expand=x[1], X_data= nadpt.X_test,Y_data=nadpt.Y_test)
    # nadpt.assesment(x_test,y_test)


