import yaml
import pickle
import os
from torch import nn

from models import *
from data import *
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%


def get_config():
    # to give me the opts
    with open("configs\\train.yaml", 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_model(iteration, model, name, opts, measures):
    network_path= os.path.join(opts['checkpoint_dir'], name+'.pkl')
    torch.save(model.state_dict(), network_path)
    pickle.dump(iteration, open("configs\\"+name+"_iter.pkl", "wb"))
    pickle.dump(measures, open("configs\\" + name + "_measures.pkl", "wb"))


def load_model(opts, name, number_of_classes):
    network_path = os.path.join(opts['checkpoint_dir'], name+'.pkl')

    if name == "complete_classification":
        net = OneHotClassify(opts, num_class=number_of_classes)
        net.image_encoder.load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage))
    elif name == "sim":
        net = SimiNet(opts)
        net.image_encoder.load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage) )

    iteration = pickle.load(open("configs\\"+name+"_iter.pkl", "rb"))
    # measures is a dict that contains
    measures = pickle.load(open("configs\\"+name+"_measures.pkl", "rb"))
    return net, iteration, measures

# =====================
# Task 1 classification
# =====================
# %%
def train_encoder_loop(opts):
    # data for training
    number_of_classes = len(os.listdir(os.getcwd() + "\\data\\augmentation\\trainset_ch\\"))

    # data for testing
    test_dataloader = test_data_loader(opts)

    if opts["encoder_load"]:
        # load model
        net, iteration, measures = load_model(opts, "complete_classification", number_of_classes)
        net = net.cuda()
        print("Load model successfully.")
    else:
        net = OneHotClassify(opts, num_class=number_of_classes).cuda()
        print("Create model successfully.")

        iteration = 0
        measures = {"accuracy": [], "loss": []}

    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=opts['SGD_lr'], momentum=opts['momentum'])
    optimizer = torch.optim.Adagrad(net.parameters(), lr=opts['opt_lr'])

    while True:
        # SGD
        iteration += 1
        running_loss = 0.0
        dataloader, number_of_classes = chinese_character_loader(opts)
        for data in dataloader:
            x, true_y = data
            # true_y??? 1.0 instead of 1
            x, true_y = x.cuda(), true_y.cuda()
            # true_y = true_y.view(-1)

            optimizer.zero_grad()
            pred_y = net(x)
            # print(pred_y.shape, true_y.shape)

            loss = criterion(pred_y, true_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("Iteration {:5d} | loss: {:6.8f}".format(iteration, running_loss))
        measures["loss"].append(running_loss)
        # Test: compute classification accuracy
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_dataloader:
                x, true_y = data
                x, true_y = x.cuda(), true_y.cuda()
                # true_y = true_y.view(-1)
                pred_y = net(x)
                _, pred_y = torch.max(pred_y, 1)

                total += x.shape[0]
                correct += (pred_y == true_y).sum().item()
            print('Accuracy: %d %%' % (100.0 * correct / total))
            measures["accuracy"].append(100.0 * correct / total)

        if iteration % opts['save_per_iter'] == 0:
            save_model(iteration, net.image_encoder, "classification", opts, measures)
            save_model(iteration, net, "complete_classification", opts, measures)
            print("Classification model saved!!!")

# =====================
# Task 2 Similar?
# =====================
# %%
def map_to_10(x):
    if x >= 0.5:
        return 1.0
    else:
        return 0.0
# %%
def train_sim_loop(opts):
    data_directory = os.getcwd() + "\\data\\augmentation\\testset_ch"
    fontPath_list = os.listdir(data_directory)  # ["a_font", "b_font", ...]
    num_fonts = len(fontPath_list)

    if opts['simi_load']:
        # load
        sim, iteration, measures = load_model(opts, "sim", num_fonts)
        sim = sim.cuda()
        print("Load sim-model successfully.")
    else:
        # generate a new model
        sim = SimiNet(opts).cuda()
        print("Create sim-model successfully.")

        iteration = 0
        measures = {"accuracy": [], "loss": []}

    if opts['simi_encoder_load']:
        # load pretrained model by classifier for the sim
        encoder_model_path = os.path.join(opts['checkpoint_dir'], 'classification.pkl')
        sim.image_encoder.load_state_dict(torch.load(encoder_model_path, map_location=lambda storage, loc: storage))
        print("Load pretrained encoder parameters for sim-model successfully.")

    if opts['fix_encoder']:
        params = list(sim.sim_model.parameters())
    else:
        params = list(sim.parameters())

    optimizer = torch.optim.Adagrad(params, lr=opts['opt_lr'])

    criterion = nn.BCELoss()

    while True:
        iteration += 1

        # Train
        running_loss = 0.0
        imgs_A_loader, imgs_B_loader, y_loader = chinese_advanced_loader(opts)

        for imgs_A, imgs_B, true_y in zip(imgs_A_loader, imgs_B_loader, y_loader):
            imgs_A, imgs_B, true_y = imgs_A.cuda(), imgs_B.cuda(), true_y.cuda()
            pred_y = sim(x_A=imgs_A, x_B=imgs_B)

            pred_y = pred_y.view(-1)
            optimizer.zero_grad()
            loss = criterion(pred_y, true_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Iteration {:5d} | loss: {:6.8f}".format(iteration, running_loss))
        measures["loss"].append(running_loss)

        # Test
        total = 0
        correct = 0
        with torch.no_grad():
            imgs_A_loader, imgs_B_loader, y_loader = chinese_advanced_loader(opts, True)

            for imgs_A, imgs_B, true_y in zip(imgs_A_loader, imgs_B_loader, y_loader):
                imgs_A, imgs_B, true_y = imgs_A.cuda(), imgs_B.cuda(), true_y.cuda()
                pred_y = sim(x_A=imgs_A, x_B=imgs_B)

                pred_y_copy = copy.copy(pred_y.data)
                pred_y_copy = pred_y_copy.cpu().data.numpy()

                pred_y_copy_ = map(map_to_10, pred_y_copy)
                pred_y_copy_ = list(pred_y_copy_)

                y_copy = copy.copy(true_y)
                y_copy = y_copy.cpu().data.numpy()

                correct += (y_copy==pred_y_copy_).sum().item()
                total += len(pred_y_copy_)

            print("Accuracy: %d %%" % (100.0 * correct/total))
            measures["accuracy"].append(100.0 * correct/total)

        if iteration % opts['save_per_iter'] == 0:
            save_model(iteration, sim, "sim", opts, measures)
            print("Simi model saved!!!")
# %%
def plot_data(measures):
    plt.figure()
    plt.plot(measures['loss'], label = "loss")
    plt.plot(measures['accuracy'], label = 'accuracy')
    plt.xlabel('iterations')
    plt.savefig('loss_acc123.png')

# %%
if __name__ == "__main__":
    opts = get_config()
    # train_encoder_loop(opts)
    # train_sim_loop(opts)
    # if need to plot then just plug in the loss to the function above
    # first model: all_losses1; second model: all_losses2
    measures = pickle.load(open("configs\\" + 'sim' + "_measures.pkl", "rb"))
    plot_data(measures)