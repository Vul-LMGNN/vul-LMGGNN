import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def encode_input(text, tokenizer):
    max_length = 512
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask

class CodeBertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=2):
        super(CodeBertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, self.nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit

class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)

        res = F.softmax(res, dim=1)

        return res


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        # self.conv.apply(init_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)
        x = self.conv(x, data.x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
