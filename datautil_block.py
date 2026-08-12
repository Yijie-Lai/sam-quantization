# This file is modified based on the ReasoningQAT and EfficientQAT projects.

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import Dataset
import os


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9

    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader



def format_openthoughts_sample(example):
    messages = []
    for item in example:
        if item["from"] == "human":
            messages.append({"role": "user", "content": item["value"]})
        elif item["from"] == "gpt":
            messages.append({"role": "assistant", "content": item["value"]})
    return {"messages": messages}


def get_openthoughts(tokenizer, train_size, val_size, seed, seqlen):
    print("get_openthoughts")

    dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    dataset = dataset.shuffle(seed=seed)
    rng = random.Random(seed)

    trainloader, valloader = [], []

    for sample in dataset:
        if len(trainloader) >= train_size and len(valloader) >= val_size:
            break

        try:
            formatted = format_openthoughts_sample(sample['conversations'])
            text = tokenizer.apply_chat_template(formatted['messages'], tokenize=False)
            enc = tokenizer(text, return_tensors='pt')

            if enc.input_ids.shape[1] >= seqlen:
                # Cover the complete response distribution instead of always
                # calibrating on the system/user prompt and response prefix.
                max_start = enc.input_ids.shape[1] - seqlen
                start = rng.randint(0, max_start)
                inp = enc.input_ids[:, start:start + seqlen]
                tar = inp.clone()

                if len(trainloader) < train_size:
                    trainloader.append((inp, tar))
                else:
                    valloader.append((inp, tar))
        except Exception:
            continue

    return trainloader, valloader


# =========================
# RedPajama concat dataset
# =========================

def get_redpajama_concat(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama_concat")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    buffer = ""
    trainloader, valloader = [], []
    rng = random.Random(seed)

    for sample in dataset:
        if len(trainloader) >= train_size and len(valloader) >= val_size:
            break

        buffer += bos + sample['text'] + eos
        enc = tokenizer(buffer, return_tensors='pt')

        if enc.input_ids.shape[1] >= seqlen:
            max_start = enc.input_ids.shape[1] - seqlen
            start = rng.randint(0, max_start)
            inp = enc.input_ids[:, start:start + seqlen]
            tar = inp.clone()

            if len(trainloader) < train_size:
                trainloader.append((inp, tar))
            else:
                valloader.append((inp, tar))

            buffer = ""

    return trainloader, valloader


def get_sweep_dataset(tokenizer, train_size, val_size, seed, seqlen, reasoning_ratio):
    print("get_sweep_dataset")

    r_train = int(train_size * reasoning_ratio)
    r_val = int(val_size * reasoning_ratio)

    nr_train = train_size - r_train
    nr_val = val_size - r_val

    train1, val1 = [], []
    train2, val2 = [], []

    if r_train > 0:
        train1, val1 = get_openthoughts(tokenizer, r_train, r_val, seed, seqlen)

    if nr_train > 0:
        train2, val2 = get_redpajama_concat(tokenizer, nr_train, nr_val, seed, seqlen)

    trainloader = train1 + train2
    valloader = val1 + val2

    rng = random.Random(seed)
    rng.shuffle(trainloader)
    rng.shuffle(valloader)

    return trainloader, valloader


def get_loaders(
    name,
    tokenizer,
    train_size=128,
    val_size=64,
    seed=0,
    seqlen=2048,
    test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)

    elif 'openthoughts' in name:
        return get_openthoughts(tokenizer, train_size, val_size, seed, seqlen)

    elif 'redpajama_concat' in name:
        return get_redpajama_concat(tokenizer, train_size, val_size, seed, seqlen)

    elif 'sweep_' in name:
        ratio = float(name.split('_')[1])
        return get_sweep_dataset(tokenizer, train_size, val_size, seed, seqlen, ratio)

    else:
        raise NotImplementedError


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'], ppl_seqlen=2048):
    results = {}

    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        if hasattr(model, 'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model, 'lm_head'):
            classifier = None
        elif hasattr(model, 'output'):
            classifier = model.output
        else:
            raise NotImplementedError

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(model.device)

            outputs = model.model(batch)

            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]

            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(shift_logits.device)

            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            nlls.append(loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()

        model.config.use_cache = use_cache

    return results


class BlockTrainDataset(Dataset):
    def __init__(
        self,
        size,
        seqlen,
        hidden_size,
        batch_size,
        dtype,
        cache_path='./cache/block_training_data',
        off_load_to_disk=False
    ):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size

        assert size % batch_size == 0

        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros(
                (self.size // self.batch_size,
                 self.batch_size,
                 self.seqlen,
                 self.hidden_size),
                dtype=self.dtype
            )

    def _initialize_data_on_disk(self):
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros(
                (self.batch_size, self.seqlen, self.hidden_size),
                dtype=self.dtype
            )
            torch.save(tensor, self._get_file_path(idx))

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        if self.off_load_to_disk:
            return torch.load(self._get_file_path(idx))
        else:
            return self.data[idx]

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            torch.save(new_data.to(self.dtype), self._get_file_path(idx))
        else:
            self.data[idx] = new_data