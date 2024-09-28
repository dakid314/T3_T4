# %%
import os
import sys

# 加载.env环境变量
if getattr(sys, "frozen", False):
    coderootdir = os.path.dirname(sys.executable)
else:
    bundle_dir = os.path.dirname(
        os.path.abspath(__file__)
    )
    coderootdir = os.path.dirname(bundle_dir)


def workdir(path: str, time: int = 2):
    originpath = path
    for _ in range(time + 1):
        if os.path.exists(os.path.join(path, '.targetdir')) == True:
            return os.path.abspath(path)
        else:
            path = os.path.join(path, '..')
    raise RuntimeError(f'Not Found targetdir: {originpath}')


proj_root_dir = workdir(path=coderootdir, time=2)

print(f"proj_root_dir: {proj_root_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='bert-diso')
    parser.add_argument('-f', type=str, required=True, help='Path to fasta.')
    parser.add_argument('-o', type=str, required=True, help='Path to Out.')
    parser.add_argument('-c', action='store_true', help='C ter.')
    args = parser.parse_args()
# %%
import torch
import numpy as np
import numpy as np
from Bio import SeqIO
import itertools
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
import torch
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, modeling_outputs
import torch
import numpy as np
import functools
import itertools
import pickle
import typing
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pandas as pd
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# %%


def load_data(path: str, label: int):
    return [
        [str(seq.seq), label]
        for seq in SeqIO.parse(path, "fasta")
    ]


# %%
sS_k = "DISO"
sS_d = {
    "-": 0,
    "*": 1,
    # "P": -100 # Padding
}
early_stop_iter = 3
max_length = 126 + 2
# %%
from functools import reduce
from typing import Optional, Union, Tuple


class CBFTC(BertForTokenClassification):

    class Custom_Classifier(torch.nn.Module):
        def __init__(self, input_seq_dim, output_seq_dim):
            super().__init__()
            self.input_seq_dim = input_seq_dim
            self.output_seq_dim = output_seq_dim
            self.output_layer = torch.nn.Sequential(
                torch.nn.LayerNorm(self.input_seq_dim),
                torch.nn.Linear(self.input_seq_dim, 1024),
                torch.nn.Sigmoid(),
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 1024),
                torch.nn.Sigmoid(),
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 1024),
                torch.nn.Sigmoid(),
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 1024),
                torch.nn.Sigmoid(),
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 1024),
                torch.nn.Sigmoid(),
                # torch.nn.Dropout(p=0.5),
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, self.output_seq_dim),
            )

        def forward(self, seq):
            logits = self.output_layer(seq)
            return logits

    def __init__(self, config):
        super().__init__(config)
        self.classifier = CBFTC.Custom_Classifier(
            input_seq_dim=config.hidden_size,
            output_seq_dim=config.num_labels
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_loss_func=None
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.TokenClassifierOutput]:
        result = super().forward(
            input_ids, attention_mask, token_type_ids, position_ids,
            head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        if custom_loss_func is None:
            return result

        result.loss = custom_loss_func(
            result.logits.view(-1, self.num_labels), labels.view(-1))
        return result


# %%
up_task_id = 'zH1SG6I9lMkhQU5R3GTv967gEWrt6KJb'
bertconfig = BertConfig.from_pretrained(
    f'{proj_root_dir}/lib/StructurePrediction/config', num_labels=len(sS_d.keys()), output_attentions=True)
upmodel = CBFTC(config=bertconfig)
upmodel.load_state_dict(torch.load(
    f"{proj_root_dir}/lib/StructurePrediction/DISO/{up_task_id}/model.pt", map_location=torch.device('cpu')))
f_tokenizer = BertTokenizer.from_pretrained(
    f'{proj_root_dir}/lib/StructurePrediction/config')

# %%


class SESeqDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_d: np.ndarray,
        transform=None,
        target_transform=None,
    ):
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform

        self.dataset_d = dataset_d
        pass

    def __len__(self):
        return len(self.dataset_d)

    def __getitem__(self, idx: int):

        seq = str(self.dataset_d[idx][0])
        label = self.dataset_d[idx][1]

        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)

        return (seq, label)


def seq_transform(seq, tokenizer, max_length):
    seq = " ".join(seq)
    return tokenizer(
        seq,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )


def label_transform(label: int):
    return torch.Tensor([label, ])


# %%
Batch_size = 256
if torch.cuda.device_count() > 1:
    Batch_size = Batch_size * torch.cuda.device_count()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


class Masked_AM_Position:
    Up_Down: str
    layer: int
    header: int


def add_mask_for_seq_start_end(mask_tensor: torch.LongTensor, device: str):
    last_one = torch.argmax((mask_tensor == 1).to(
        dtype=torch.int) * torch.arange(0, mask_tensor.shape[1], ).to(device), dim=-1)

    mask_tensor[torch.BoolTensor([True] * mask_tensor.shape[0]), last_one] = 0
    mask_tensor[:, 0] = 0

    return mask_tensor


def eval_fn(
    data_loader,
    upstream_model,
    device,
    mask_am_infor: Masked_AM_Position,
    mask_seq_infor
):

    upstream_model.eval()
    if torch.cuda.device_count() > 1:
        upstream_model = torch.nn.DataParallel(upstream_model)

    predictions = np.array([], dtype=np.float64).reshape(
        0,
        max_length,
        len(sS_d.keys())
    )

    # 检查Mask 信息
    assert (
        mask_am_infor is None
    ) or (
        mask_am_infor.Up_Down == 'Up' and mask_am_infor.header is not None and mask_am_infor.layer is not None
    ) or (
        mask_am_infor.Up_Down == 'Down' and mask_am_infor.header is not None
    )
    assert (
        mask_seq_infor is None
    ) or (
        max(mask_seq_infor) < max_length - 2
    )

    # 处理 Up Head Mask
    up_head_mask = torch.ones(
        (
            upstream_model.config.num_hidden_layers,
            upstream_model.config.num_attention_heads
        ),
        dtype=torch.long
    )
    up_head_mask = up_head_mask.to(device)
    down_head_mask = None

    # Mask Do it!
    if mask_am_infor is not None:
        if mask_am_infor.Up_Down == "Up":
            up_head_mask[mask_am_infor.layer][mask_am_infor.header] = 0

        elif mask_am_infor.Up_Down == "Down":
            down_head_mask = [mask_am_infor.header, ]
            # down_head_mask =  down_head_mask.to(device)

    with torch.no_grad():
        for index, dataset in enumerate(data_loader):

            batch_input_ids = torch.squeeze(
                dataset[0]['input_ids'].to(device, dtype=torch.long), dim=1)
            batch_tok_type_id = torch.squeeze(
                dataset[0]['token_type_ids'].to(device, dtype=torch.long), dim=1)
            batch_att_mask = torch.squeeze(
                dataset[0]['attention_mask'].to(device, dtype=torch.long), dim=1)

            ss_output = upstream_model(
                batch_input_ids,
                token_type_ids=batch_tok_type_id,
                attention_mask=batch_att_mask,
                head_mask=up_head_mask
            )

            predictions = np.concatenate(
                (predictions, ss_output.logits),
                axis=0
            )

    return predictions


# %%


class Eval_Result:
    loss: float
    predictions: np.ndarray
    true_labels: np.ndarray
    down_attention_map: np.ndarray
    up_attention_map: np.ndarray
    up_attention_map_layer0: np.ndarray
    mask_infor: dict


def train_engine(
    upstream_model,
    valid_dsl,
    device,
    mask_am_infor: Masked_AM_Position = None,
    mask_seq_infor=None
):
    upstream_model = upstream_model.to(device)

    result = Eval_Result()

    result.predictions = eval_fn(
        data_loader=valid_dsl,
        upstream_model=upstream_model,
        device=device,
        mask_am_infor=mask_am_infor,
        mask_seq_infor=mask_seq_infor
    )

    result.mask_infor = mask_am_infor.__dict__ if mask_am_infor is not None else None

    return result


# %%
if __name__ == "__main__":

    test_ds = SESeqDataset(
        dataset_d=[
            [str(item.seq) if args.c == False else str(item.seq)[::-1], -100]
            for item in list(SeqIO.parse(args.f, "fasta"))
        ],
        transform=functools.partial(
            seq_transform,
            tokenizer=f_tokenizer,
            max_length=max_length
        ),
        target_transform=functools.partial(
            label_transform
        )
    )
    test_dsl = DataLoader(
        test_ds,
        batch_size=Batch_size,
        shuffle=False
    )
    result = train_engine(
        upstream_model=upmodel,
        valid_dsl=test_dsl,
        device=device,
    )

    os.makedirs(
        os.path.dirname(args.o), exist_ok=True
    )

    for index, item in enumerate(list(SeqIO.parse(args.f, "fasta"))):
        result.predictions[index, len(item.seq) + 1:, :] = 0
    result.predictions = result.predictions[:, 1:-1, :]

    with open(args.o, "bw+") as f:
        pickle.dump(
            {
                "cter": args.c,
                "type": "diso",
                "seq_id": [
                    item.id
                    for item in list(SeqIO.parse(args.f, "fasta"))
                ],
                "value": result.predictions
            },
            f
        )
