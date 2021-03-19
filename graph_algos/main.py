import os
import argparse
import pickle
import random
import numpy as np
import shutil
import torch
import networkx as nx
from torch.autograd import Variable

from models.gcnn import GGNN
from models.gcnn_edges import GGNN_E
from util import *
import logging
import math

random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

best_prec, best_loss, best_model_test_acc, best_model_test_loss, best_model_mape_loss = 0.0, 1e+8 * 1.0, 0.0, 1e+8 * 1.0, 1e+8 * 1.0
is_best = False
best_epoch = 0

model_types = {'GGNN': GGNN, 'GGNN_E': GGNN_E}


def save_checkpoint(state, is_best, epoch, args):
    if not is_best:
        return
    """Saves checkpoint to disk"""

    directory = "models_dir/Train_%s/Test_%s/Task%s/%s/" % (args.train, args.test, args.subtype, args.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'model_best.pth.tar'
    torch.save(state, filename)


def cvt_data_axis(dataset):
    data = []
    label = []
    for d, ans in dataset:
        data.append(d)
        label.append(ans)
    return (data, label)


def tensor_data(data, i, args):
    nodes = data[0][args.batch_size * i:args.batch_size * (i + 1)]
    if args.loss_fn == 'cls':
        ans = torch.LongTensor(data[1][args.batch_size * i:args.batch_size * (i + 1)]).to(args.device)
    else:
        ans = torch.FloatTensor(data[1][args.batch_size * i:args.batch_size * (i + 1)]).to(args.device)
    return nodes, ans


def train(epoch, dataset, args, model):
    model.train()
    train_size = len(dataset)
    bs, subtype = args.batch_size, args.subtype
    random.shuffle(dataset)

    data = cvt_data_axis(dataset)
    running_loss, running_loss_mape = 0.0, 0.0
    accuracys = []
    losses, losses_mape = [], []
    batch_runs = max(1, train_size // bs)
    for batch_idx in range(batch_runs):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.train_(input_nodes, label)
        running_loss += loss.item()
        running_loss_mape += mape_loss.item()

        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)

        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)] Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(
                epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, subtype, accuracy,
                running_loss / (1 * args.log_interval), running_loss_mape / (1 * args.log_interval)))
            logging.info(
                'Train Epoch: {} [{}/{} ({:.2f}%)] Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(
                    epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, subtype, accuracy,
                    running_loss / (1 * args.log_interval), running_loss_mape / (1 * args.log_interval)))
            running_loss, running_loss_mape = 0.0, 0.0

    avg_accuracy = sum(accuracys) * 1.0 / len(accuracys)
    avg_losses = sum(losses) * 1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) * 1.0 / len(losses_mape)

    print('\nTrain set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(subtype, avg_accuracy,
                                                                                         avg_losses, avg_losses_mape))
    logging.info('\nTrain set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(subtype, avg_accuracy,
                                                                                                avg_losses,
                                                                                                avg_losses_mape))


def validate(epoch, dataset, args, model):
    global is_best, best_prec, best_loss

    model.eval()
    test_size = len(dataset)
    bs, subtype = args.batch_size, args.subtype
    data = cvt_data_axis(dataset)

    accuracys = []
    losses, losses_mape = [], []
    for batch_idx in range(test_size // bs):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)

    avg_accuracy = sum(accuracys) * 1.0 / len(accuracys)
    avg_losses = sum(losses) * 1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) * 1.0 / len(losses_mape)
    print('Validation set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(subtype, avg_accuracy,
                                                                                            avg_losses,
                                                                                            avg_losses_mape))
    logging.info(
        'Validation set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f}'.format(subtype, avg_accuracy,
                                                                                          avg_losses, avg_losses_mape))

    if args.loss_fn == 'cls':
        is_best = avg_accuracy > best_prec
    else:  # lif args.loss_fn == 'reg':
        is_best = avg_losses < best_loss
    best_prec = max(avg_accuracy, best_prec)
    best_loss = min(avg_losses, best_loss)


def test(epoch, dataset, args, model):
    global is_best, best_model_test_acc, best_model_test_loss, best_epoch, best_model_mape_loss

    model.eval()
    test_size = len(dataset)
    bs, subtype = args.batch_size, args.subtype
    data = cvt_data_axis(dataset)

    accuracys = []
    losses, losses_mape = [], []
    for batch_idx in range(test_size // bs):
        input_nodes, label = tensor_data(data, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)

        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)

    avg_accuracy = sum(accuracys) * 1.0 / len(accuracys)
    avg_losses = sum(losses) * 1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) * 1.0 / len(losses_mape)

    print('Test set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f} \n'.format(subtype, avg_accuracy,
                                                                                         avg_losses, avg_losses_mape))
    logging.info('Test set: Subtype {} accuracy: {:.2f}%, loss: {:.7f}, mape: {:.7f} \n'.format(subtype, avg_accuracy,
                                                                                                avg_losses,
                                                                                                avg_losses_mape))

    if is_best:
        best_model_test_acc = avg_accuracy
        best_model_test_loss = avg_losses
        best_model_mape_loss = avg_losses_mape
        best_epoch = epoch

    if epoch % 10 == 0:
        print(
            '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(
                best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))
        logging.info(
            '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(
                best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))


def data_portion(args, train_datasets, test_datasets, validation_datasets):
    if args.portion < 1.0:
        train_size = int(math.floor(len(train_datasets) * args.portion))
        np.random.seed(random_seed)
        indices = np.random.choice(len(train_datasets), train_size)
        print(indices[0:10])
        logging.info(indices[0:10])
        train_datasets = [train_datasets[i] for i in indices]
    return train_datasets, test_datasets, validation_datasets


def load_data(index_filename, mode):
    # print(index_filename)
    with open("./run/%s.txt" % index_filename, 'r') as f:
        data = []
        for line in f:
            # print(line)
            with open("%s" % line.strip(), 'rb') as f2:
                data.extend(pickle.load(f2)[mode])

    # s2vs = []
    # for g, ans in data:
    #     neighbors = []
    #     node_features = []
    #     for i in sorted(list(g.nodes())):
    #         neighbors.append(list(g.neighbors(i)))
    #         node_features.append(g.nodes[i]['node_features'])
    #     node_features = torch.Tensor(np.array(node_features))
    #     s2vg = S2VGraph(ans, node_features, neighbors, g)
    #     s2vs.append((s2vg, ans))
    # return s2vs
    return data

def load_aw(index_filename, subtype):
    with open("./%s.txt" % index_filename, 'r') as f:
        for line in f:
            with open("%s" % line.strip(), 'rb') as f2:
                train_datasets, test_datasets, validation_datasets = pickle.load(f2, encoding="bytes")
    datas = [train_datasets, test_datasets, validation_datasets]
    data_graphs = []
    for datasets in datas:
        graphs = []
        for data in datasets:
            node_features = torch.Tensor(data[0])
            label = data[1][subtype]
            n_objects = 25
            graph = nx.complete_graph(n_objects)
            neighbors = [list(range(n_objects))] * n_objects
            graph = S2VGraph(label, node_features, neighbors, graph)
            graphs.append((graph, label))
        data_graphs.append(graphs)
    return data_graphs[0], data_graphs[1], data_graphs[2]


def setup_logs(args):
    file_dir = "results"
    if not args.no_log:
        files_dir = '%s/Train_%s/Test_%s/Task%s/sampling%s' % (
        file_dir, args.train, args.test, args.subtype, args.portion)
        args.files_dir = files_dir
        if args.model in ['GGNN', 'GGNN_E']:
            args.filename = '%s_%s_lr%s_hdim%s_fc%s_mlp%s_%s_%s_bs%s_epoch%d_seed%d.log' \
                            % (args.model, args.n_iter, args.lr, args.hidden_dim, args.fc_output_layer, args.mlp_layer,
                               args.graph_pooling_type, args.neighbor_pooling_type, args.batch_size, args.epochs,
                               random_seed)

        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
        mode = 'w+'
        if args.resume:
            mode = 'a+'
        logging.basicConfig(format='%(message)s',
                            level=logging.INFO,
                            datefmt='%m-%d %H:%M',
                            filename="%s/%s" % (args.files_dir, args.filename),
                            filemode='w+')

        print(vars(args))
        logging.info(vars(args))


def resume(args, model):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        logging.info("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        best_model_test_acc = checkpoint['best_model_test_acc']
        best_model_test_loss = checkpoint['best_model_test_loss']
        best_model_mape_loss = checkpoint['best_model_mape_loss']
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model


def main():
    parser = argparse.ArgumentParser(description='PyTorch Reasoning2')

    # Model specifications
    parser.add_argument('--model', type=str, choices=['GGNN_E', 'GGNN'], default='GGNN_E', help='choose which model')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh', 'linear', 'sigmoid'], default='relu',
                        help='activation function')
    parser.add_argument('--n_iter', type=int, default=1, help='number of RN/RRN iterations/layers (default: 1)')
    parser.add_argument('--mlp_layer', type=int, default=2, help='number of layers for MLPs in RN/RRN/MLP (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='feature hidden dimension of MLPs (default: 128)')
    parser.add_argument('--fc_output_layer', type=int, default=1,
                        help='number of layers for output(softmax) MLP in RN/RRN/MLP (default: 3)')
    parser.add_argument('--graph_pooling_type', type=str, default="max", choices=["sum", "mean", "max", "min"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "mean", "max", "min"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--add_self_loop', action='store_true',
                        default=False, help='add self loops in case graph does not contain it')

    # Training settings
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, help='resume from model stored')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--loss_fn', type=str, choices=['cls', 'reg', 'mape'], default='reg',
                        help='classification or regression loss')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='Adam or SGD')

    # Logging and storage settings
    parser.add_argument('--log_file', type=str, default='accuracy.log', help='dataset filename')
    parser.add_argument('--save_model', action='store_true', default=False, help='store the training models')
    parser.add_argument('--no_log', action='store_true', default=False, help='disables logging of results')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--filename', type=str, default='', help='the file which store trained model logs')
    parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')

    # Data settings
    parser.add_argument('--train', type=str, default='maxdeg', help='train index filename')
    parser.add_argument('--val', type=str, default=None, help='test index filename')
    parser.add_argument('--test', type=str, default=None, help='test index filename')
    parser.add_argument('--portion', type=float, default=1.0, help='sampling portion')
    parser.add_argument('--no_coord', action='store_true',
                        default=False, help='disables coordinate embedding in object representation')
    parser.add_argument('--weight', type=str, default=None, help='the directory to store trained models')
    parser.add_argument('--edge_feature_size', type=int, default=1, help='size of edge features')
    parser.add_argument('--node_feature_size', type=int, default=4, help='size of node features')
    parser.add_argument('--num_colors', default=20, type=int, help='num of possible colors for each node')
    parser.add_argument('--num_levels', default=10, type=int, help='num of possible levels for each node')
    parser.add_argument('--subtype', type=int, default=8, help='question subtypes we want to test')

    # other settings
    parser.add_argument('--return_correct', action='store_true', default=False, help='return correct indices')

    # debug settings
    parser.add_argument('--debug', action='store_true', default=False, help='return correct indices')
    parser.add_argument('--print_debug', action='store_true', default=False, help='return correct indices')

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.val == None:
        args.val = args.train
    if args.test == None:
        args.test = args.train

    print('begin load data', args.subtype, args.subtype == 8)
    if args.debug:
        train_datasets, test_datasets, validation_datasets = load_aw(args.train, args.subtype)
    elif args.subtype in [8, 14]:
        with open("./run/%s.txt" % args.train, 'r') as f:
            for line in f:
                with open("%s" % line.strip(), 'rb') as f2:
                    train_datasets, validation_datasets, test_datasets = pickle.load(f2)
    else:
        train_datasets = load_data(args.train, 0)
        validation_datasets = load_data(args.train, 1)
        test_datasets = load_data(args.test, 2)
    print('end load data')

    args.node_feature_size = len(train_datasets[0][0].node_features[0])
    setup_logs(args)

    model = model_types[args.model](args).to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.5)

    subtype, bs = args.subtype, args.batch_size

    model_dirs = './model_dirs'
    try:
        os.makedirs(model_dirs)
    except:
        print('directory {} already exists'.format(model_dirs))

    if args.epochs == 0:
        epoch = 0
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        args.epochs = -1

    for epoch in range(1, args.epochs + 1):
        train(epoch, train_datasets, args, model)
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        scheduler.step()
        if is_best and args.save_model:  # epoch%args.log_interval == 0
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'args': args,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'best_model_test_acc': best_model_test_acc,
                'best_model_test_loss': best_model_test_loss,
                'best_model_mape_loss': best_model_mape_loss,
                'optimizer': model.optimizer.state_dict(),
            }, is_best, epoch, args)

    print(
        '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(
            best_model_test_acc, best_model_test_loss, best_epoch))
    logging.info(
        '************ Best model\'s test accuracy: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(
            best_model_test_acc, best_model_test_loss, best_epoch))


if __name__ == '__main__':
    main()
