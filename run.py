import argparse
import gc
import shutil
from argparse import ArgumentParser
import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg as cpg
import torch
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from test import test

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    #print(len(result))
    #result = result.iloc[11001:]
    #print(len(result))
    result = result.head(200)

    return result

def CPG_generator():
    """
    Generates Code Property Graph (CPG) datasets from raw data.

    :return: None
    """
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)

    # Here, taking the Devign dataset as an example,
    # specific modifications need to be made according to different dataset formats.
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()

def Embed_generator():
    """
    Generates embeddings from Code Property Graph (CPG) datasets.

    :return: None
    """
    context = configs.Embed()
    dataset_files = data.get_directory_files(PATHS.cpg)

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")

        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        cpg_dataset["input"] = cpg_dataset.apply(lambda row: process.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                            context.edge_type), axis=1)
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        # write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        # write(cpg_dataset[["input", "target","func"]], PATHS.input, f"{file_name}_{FILES.input}")
        data.write(cpg_dataset[["input", "target", "func"]], PATHS.input, f"{file_name}_{FILES.input}")

        del cpg_dataset
        gc.collect()


def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model using the provided data.

    :param model: The model to be trained.
    :param device: The device to perform training on (e.g., 'cpu' or 'cuda').
    :param train_loader: The data loader containing the training data.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :return: None
    """

    model.train()
    best_acc = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)

        y_pred = model(batch)
        model.zero_grad()
        # print("y_pred data type:", y_pred.dtype)
        # print("batch.y.squeeze() data type:", batch.y.squeeze().dtype)
        batch.y = batch.y.squeeze().long()
        loss = F.cross_entropy(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(batch),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
def validate(model, device, test_loader):
    """
    Validates the model using the provided test data.

    :param model: The model to be validated.
    :param device: The device to perform validation on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch)

        batch.y = batch.y.squeeze().long()
        test_loss += F.cross_entropy(y_, batch.y).item()
        pred = y_.max(-1, keepdim=True)[1]

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'], yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('-cpg', '--cpg', action='store_true', help='Specify to perform CPG generation task')
    parser.add_argument('-embed', '--embed', action='store_true', help='Specify to perform Embedding generation task')
    parser.add_argument('-mode', '--mode', default="train", help='Specify the mode (e.g., train, test)')
    parser.add_argument('-path', '--path', default=None, help='Specify the path for the model')

    args = parser.parse_args()

    if args.cpg:
        CPG_generator()
    if args.embed:
        Embed_generator()

    context = configs.Process()
    input_dataset = loads(PATHS.input)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            train_val_test_split(input_dataset, shuffle=context.shuffle)))

    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=Bertggnn.learning_rate, weight_decay=Bertggnn.weight_decay)

        best_acc = 0.0
        NUM_EPOCHS = context.epochs
        PATH = args.path
        for epoch in range(1, NUM_EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch)
            acc, precision, recall, f1 = validate(model, DEVICE, val_loader)
            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), PATH)
            print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

    model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to("cuda")
    model_test.load_state_dict(torch.load(args.path))
    accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader)


