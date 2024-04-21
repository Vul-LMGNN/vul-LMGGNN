import json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
torch.manual_seed(2020)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = th.device('cpu')
gpu = th.device('cuda:0')
def process_json_file(file_path):
    """
    Processes a JSON file and converts it into a pandas DataFrame.

    :param file_path: The path to the JSON file.
    :return: A pandas DataFrame containing the processed data.
    """
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            # Process each line of JSON data
            # Perform your operations, e.g., extracting required fields or executing other logic
            # Here, let's assume you extract some fields from the JSON data and store them in a dictionary
            data_list.append(json_line)
    df = pd.DataFrame(data_list)
    return df

def encode_input(text, tokenizer):
    """
    Encodes the input text using the provided tokenizer.
    :param text: The input text to be encoded.
    :param tokenizer: The tokenizer object.
    :return: Tuple containing input_ids and attention_mask tensors.
    """
    max_length = 128
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')

    return input.input_ids, input.attention_mask

class CustomDataset(Data.Dataset):
    def __init__(self, input_ids, attention_mask, label, nums):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.num = nums

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.label[index], self.num[index]


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(BertClassifier, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output.to(device))
        logits = self.fc(x)

        return logits

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))


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
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        y_pred = model(x1, x2)
        model.zero_grad()
        loss = F.cross_entropy(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))

def val(model, device, test_loader):
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
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        with torch.no_grad():
            y_ = model(x1, x2)

        test_loss += F.cross_entropy(y_, y.squeeze()).item()

        pred = y_.max(-1, keepdim=True)[1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1


def test(model, device, test_loader):
    """
    Tests the model using the provided test data.

    :param model: The model to be tested.
    :param device: The device to perform testing on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    num_list = []

    for batch_idx, batch in enumerate(test_loader):
        x1, x2, y, num = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        with torch.no_grad():
            y_ = model(x1, x2)

        test_loss += F.cross_entropy(y_, y.squeeze()).item()
        pred = y_.max(-1, keepdim=True)[1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])

        num_list.extend(num.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')

    # Save predicted results, true labels, predicted probabilities, and num attribute to a txt file
    results_array = np.column_stack((y_true, y_pred, y_probs, num_list))  # Append num_list to results_array
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt('results.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

if __name__ == '__main__':

    # Call the function to process a large JSON file and return a Pandas DataFrame
    df = process_json_file('dataset/DisverseVul/diversevul_20230702.json')
    filtered_df = df[df['cwe'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    shuffled_df = filtered_df.sample(frac=1, random_state=78).reset_index(drop=True)
    input_dataset = shuffled_df[['target', 'func', 'cwe']]

    train_size = 201238
    validate_size = 40248
    test_size = 40248
    col = 0

    # Split the dataset into train, validate, and test sets
    y_train = torch.tensor(input_dataset.iloc[:train_size, col].values).to(torch.int64)
    y_test = torch.tensor(input_dataset.iloc[train_size + validate_size:train_size + validate_size + test_size, col].values).to(torch.int64)
    y_validate = torch.tensor(input_dataset.iloc[train_size:train_size + validate_size, col].values).to(torch.int64)

    total = train_size + validate_size + test_size
    data = input_dataset["func"].tolist()[:total]
    num = input_dataset.index.tolist()

    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids_, attention_mask_ = encode_input(data, tokenizer)

    input_ids, attention_mask, nums = {}, {}, {}
    input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[:train_size], input_ids_[
                                                                                       train_size:train_size + validate_size], input_ids_[
                                                                                                                               -test_size:]
    attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[
                                                                             :train_size], attention_mask_[
                                                                                           train_size:train_size + validate_size], attention_mask_[
                                                                                                                                   -test_size:]
    nums['train'], nums['val'], nums['test'] = num[:train_size], num[train_size:train_size + validate_size], num[
                                                                                                             -test_size:]

    input_ids['train'] = input_ids['train'].to(device)
    input_ids['val'] = input_ids['val'].to(device)
    input_ids['test'] = input_ids['test'].to(device)
    attention_mask['train'] = attention_mask['train'].to(device)
    attention_mask['val'] = attention_mask['val'].to(device)
    attention_mask['test'] = attention_mask['test'].to(device)

    label = {}
    label['train'], label['val'], label['test'] = y_train, y_validate, y_test
    label['train'] = label['train'].to(device)
    label['val'] = label['val'].to(device)
    label['test'] = label['test'].to(device)

    datasets = {}
    loader = {}

    for split in ['train', 'val', 'test']:
        datasets[split] = CustomDataset(input_ids[split], attention_mask[split], label[split], nums[split])
        loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

    model = BertClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 10
    PATH = 'Bert(Diverse)_'

    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, loader["train"], optimizer, epoch)
        acc, precision, recall, f1 = val(model, device, loader["val"])
        model_name = PATH + f'epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_name)

        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

    # Test the model
    model_test = BertClassifier().to(device)
    model_test.load_state_dict(torch.load("Bert(Diverse)_epoch_7.pth"))
    accuracy, precision, recall, f1 = test(model_test, device, loader['test'])
