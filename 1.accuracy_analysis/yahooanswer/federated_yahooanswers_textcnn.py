import argparse
import csv
import hashlib
import random
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.TextCNN import TextCNN


TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
YAHOO_ANSWERS_TRAIN_URL = (
    "https://huggingface.co/datasets/yassiracharki/Yahoo_Answers_10_categories_for_NLP/resolve/main/train.csv"
)
YAHOO_ANSWERS_TEST_URL = (
    "https://huggingface.co/datasets/yassiracharki/Yahoo_Answers_10_categories_for_NLP/resolve/main/test.csv"
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_MAX_RETRIES = 5
NUM_CLASSES = 10
YAHOO_ANSWERS_LABELS = [
    "society_culture",
    "science_math",
    "health",
    "education_reference",
    "computers_internet",
    "sports",
    "business_finance",
    "entertainment_music",
    "family_relationships",
    "politics_government",
]
TOKEN_PATTERN = re.compile(r"(https?://\S+)|(@\w+)|(#\w+)|([A-Za-z0-9_']+)|([^\w\s])")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple federated learning on Yahoo Answers Topics with TextCNN and Dirichlet partitioning"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "yahoo_answers_topics")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument(
        "--private-train-size",
        type=int,
        default=500000,
        help="Balanced private training pool size sampled from the full Yahoo Answers dataset",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=100000,
        help="Balanced evaluation subset size sampled from the full Yahoo Answers dataset",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--min-samples-per-client", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-channels", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    if torch.cuda.is_available() and 0 <= args.gpu_id < torch.cuda.device_count():
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")


def _build_download_request(url, start_byte=0):
    headers = {"User-Agent": "Mozilla/5.0"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    return urllib.request.Request(url, headers=headers)


def _stream_response_to_file(response, temp_path, label, start_byte, total_size):
    bytes_downloaded = start_byte
    last_reported_percent = (start_byte * 100 // total_size) if total_size and total_size > 0 else -1
    write_mode = "ab" if start_byte > 0 else "wb"

    with temp_path.open(write_mode) as handle:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            bytes_downloaded += len(chunk)

            if total_size and total_size > 0:
                percent = min(int(bytes_downloaded * 100 / total_size), 100)
                if percent != last_reported_percent and percent % 10 == 0:
                    print(f"{label} download progress: {percent}%")
                    last_reported_percent = percent

    return bytes_downloaded


def _download_file(url, target_path, label):
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    start_byte = temp_path.stat().st_size if temp_path.exists() else 0

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            if start_byte > 0:
                print(
                    f"Resuming {label} to {target_path} from {start_byte} bytes "
                    f"(attempt {attempt}/{DOWNLOAD_MAX_RETRIES}) ..."
                )
            else:
                print(f"Downloading {label} to {target_path} (attempt {attempt}/{DOWNLOAD_MAX_RETRIES}) ...")

            request = _build_download_request(url, start_byte=start_byte)
            with urllib.request.urlopen(request, timeout=60) as response:
                status_code = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range")
                content_length = response.headers.get("Content-Length")
                response_length = int(content_length) if content_length is not None else None
                supports_resume = status_code == 206 or content_range is not None

                if start_byte > 0 and not supports_resume:
                    print(f"{label} server did not honor resume request. Restarting from 0%.")
                    if temp_path.exists():
                        temp_path.unlink()
                    start_byte = 0
                    continue

                if supports_resume and response_length is not None:
                    total_size = start_byte + response_length
                else:
                    total_size = response_length

                bytes_downloaded = _stream_response_to_file(
                    response,
                    temp_path,
                    label,
                    start_byte,
                    total_size,
                )

            if total_size is not None and bytes_downloaded < total_size:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got only {bytes_downloaded} out of {total_size} bytes",
                    None,
                )

            temp_path.replace(target_path)
            print(f"{label} download finished: {target_path}")
            return
        except Exception as exc:
            if attempt >= DOWNLOAD_MAX_RETRIES:
                raise RuntimeError(
                    f"{label} download failed after {DOWNLOAD_MAX_RETRIES} attempts. "
                    f"Partial file kept at {temp_path} for resume."
                ) from exc

            retained_size = temp_path.stat().st_size if temp_path.exists() else 0
            print(
                f"{label} download interrupted on attempt {attempt}/{DOWNLOAD_MAX_RETRIES}: {exc}. "
                f"Retrying from {retained_size} bytes ..."
            )
            start_byte = retained_size
            time.sleep(min(2 * attempt, 10))


def maybe_download_yahooanswers(data_dir, train_csv, test_csv, should_download):
    if train_csv.exists() and test_csv.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    if not should_download:
        print("Yahoo Answers Topics csv files were not found locally. Trying automatic download now.")
    else:
        print("Yahoo Answers Topics automatic download requested. Starting download.")

    try:
        if not train_csv.exists():
            _download_file(YAHOO_ANSWERS_TRAIN_URL, train_csv, "Yahoo Answers Topics train.csv")
        if not test_csv.exists():
            _download_file(YAHOO_ANSWERS_TEST_URL, test_csv, "Yahoo Answers Topics test.csv")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Yahoo Answers Topics automatically into {data_dir}. "
            f"Please place {TRAIN_FILE_NAME} and {TEST_FILE_NAME} there manually if needed."
        ) from exc

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Yahoo Answers Topics download completed, but expected {train_csv} and {test_csv} were not found."
        )


def normalize_label(raw_label):
    raw_label = str(raw_label).strip().strip('"')
    if not raw_label.isdigit():
        return None
    label = int(raw_label) - 1
    if 0 <= label < NUM_CLASSES:
        return label
    return None


def load_yahooanswers_csv(csv_path, max_samples):
    samples = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            label = normalize_label(row[0])
            if label is None:
                continue
            text = " ".join(cell.strip() for cell in row[1:] if cell.strip())
            if not text:
                continue
            samples.append((text, label))
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples


def _balanced_class_quotas(total_size, class_ids):
    num_classes = len(class_ids)
    base_quota = total_size // num_classes
    remainder = total_size % num_classes
    return {
        int(class_id): base_quota + (1 if offset < remainder else 0)
        for offset, class_id in enumerate(class_ids)
    }


def balanced_two_way_split_indices(targets, private_train_size, eval_size, seed):
    targets = np.asarray(targets, dtype=np.int64)
    class_ids = sorted(np.unique(targets).tolist())
    rng = np.random.default_rng(seed)

    requested_sizes = {
        "private_train": int(private_train_size),
        "eval": int(eval_size),
    }
    if any(size <= 0 for size in requested_sizes.values()):
        raise ValueError(
            "private_train_size and eval_size must both be positive "
            f"for balanced Yahoo Answers sampling: {requested_sizes}"
        )

    quotas = {
        split_name: _balanced_class_quotas(split_size, class_ids)
        for split_name, split_size in requested_sizes.items()
    }

    split_indices = {split_name: [] for split_name in requested_sizes}
    for class_id in class_ids:
        class_positions = np.where(targets == class_id)[0]
        rng.shuffle(class_positions)
        required_for_class = quotas["private_train"][int(class_id)] + quotas["eval"][int(class_id)]
        if required_for_class > len(class_positions):
            raise ValueError(
                f"Not enough Yahoo Answers samples for class {class_id}. "
                f"Need {required_for_class}, but only found {len(class_positions)}."
            )

        private_take = quotas["private_train"][int(class_id)]
        eval_take = quotas["eval"][int(class_id)]
        split_indices["private_train"].extend(class_positions[:private_take].tolist())
        split_indices["eval"].extend(class_positions[private_take:private_take + eval_take].tolist())

    for split_name in split_indices:
        rng.shuffle(split_indices[split_name])

    return (
        np.asarray(split_indices["private_train"], dtype=np.int64),
        np.asarray(split_indices["eval"], dtype=np.int64),
    )


def tokenize_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(text)]
    return tokens or ["<empty>"]


def stable_hash_token(token, vocab_size):
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (vocab_size - 2) + 2


def encode_text(text, vocab_size, max_length):
    tokens = tokenize_text(text)[:max_length]
    token_ids = [stable_hash_token(token, vocab_size) for token in tokens]
    if len(token_ids) < max_length:
        token_ids.extend([0] * (max_length - len(token_ids)))
    return token_ids


class YahooAnswersTopicsDataset(Dataset):
    def __init__(self, samples, vocab_size, max_length):
        self.samples = list(samples)
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        token_ids = encode_text(text, self.vocab_size, self.max_length)
        return torch.tensor(token_ids, dtype=torch.long), int(label)


def build_loader_kwargs(args, shuffle):
    use_cuda = torch.cuda.is_available()
    kwargs = {
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def dirichlet_partition(targets, num_clients, alpha, min_samples_per_client, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for class_id in classes:
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        client_sizes = [len(indices) for indices in client_indices]
        if min(client_sizes) >= min_samples_per_client:
            break

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def build_client_loaders(train_dataset, client_indices, batch_size, loader_kwargs):
    client_loaders = []
    client_sizes = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, **loader_kwargs))
        client_sizes.append(len(indices))
    return client_loaders, client_sizes


def build_model(args):
    return TextCNN(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_classes=NUM_CLASSES,
        num_channels=args.num_channels,
        dropout=args.dropout,
    )


def clone_state_dict_to_cpu(state_dict):
    return OrderedDict((key, value.detach().clone().cpu()) for key, value in state_dict.items())


def train_local_model(global_state, data_loader, sample_count, device, args, local_seed):
    model = build_model(args).to(device)
    model.load_state_dict(global_state)
    model.train()

    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.local_lr, weight_decay=args.weight_decay)
    for _ in range(args.local_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

    return clone_state_dict_to_cpu(model.state_dict()), sample_count


def aggregate_states(state_dicts, weights):
    total_weight = float(sum(weights))
    aggregated = OrderedDict()
    for key in state_dicts[0].keys():
        reference = state_dicts[0][key]
        if reference.is_floating_point():
            value = torch.zeros_like(reference)
            for state_dict, weight in zip(state_dicts, weights):
                value = value + state_dict[key] * (weight / total_weight)
            aggregated[key] = value
        else:
            aggregated[key] = reference.clone()
    return aggregated


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def print_first_round_distribution(targets, selected_clients, client_indices):
    print("First-round client data distribution:")
    for client_id in selected_clients:
        client_targets = np.asarray(targets[client_indices[client_id]], dtype=np.int64)
        class_counts = np.bincount(client_targets, minlength=NUM_CLASSES)
        total_samples = int(class_counts.sum())
        print(f"  Client {client_id}: total_samples={total_samples}")
        for class_id, count in enumerate(class_counts):
            ratio = (count / total_samples * 100.0) if total_samples > 0 else 0.0
            print(f"    class {class_id}: count={int(count)} ratio={ratio:.2f}%")


def run_federated_learning(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be less than or equal to num_clients")

    set_seed(args.seed)
    device = resolve_device(args)

    train_csv = args.train_csv or (args.data_dir / TRAIN_FILE_NAME)
    test_csv = args.test_csv or (args.data_dir / TEST_FILE_NAME)
    maybe_download_yahooanswers(args.data_dir, train_csv, test_csv, args.download)

    train_samples = load_yahooanswers_csv(train_csv, args.max_train_samples)
    test_samples = load_yahooanswers_csv(test_csv, args.max_test_samples)
    if not train_samples or not test_samples:
        raise RuntimeError("Failed to load Yahoo Answers Topics samples from the provided csv files.")

    all_samples = train_samples + test_samples
    all_targets = np.asarray([label for _, label in all_samples], dtype=np.int64)
    full_dataset = YahooAnswersTopicsDataset(all_samples, args.vocab_size, args.max_length)

    private_indices, eval_indices = balanced_two_way_split_indices(
        all_targets,
        args.private_train_size,
        args.eval_size,
        args.seed,
    )
    private_targets = all_targets[private_indices]

    client_indices = dirichlet_partition(
        private_targets,
        args.num_clients,
        args.dirichlet_alpha,
        args.min_samples_per_client,
        args.seed,
    )

    train_loader_kwargs = build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = build_loader_kwargs(args, shuffle=False)
    client_loaders, client_sizes = build_client_loaders(
        full_dataset,
        client_indices,
        args.batch_size,
        train_loader_kwargs,
    )
    eval_subset = Subset(full_dataset, eval_indices.tolist())
    test_loader = DataLoader(eval_subset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    print(
        "Dataset setup:",
        "dataset=YahooAnswersTopics",
        f"private_train_size={len(private_indices)}",
        f"eval_size={len(eval_indices)}",
        f"num_classes={NUM_CLASSES}",
        f"vocab_size={args.vocab_size}",
        f"max_length={args.max_length}",
        "split_strategy=balanced_class_sampling",
    )
    print(
        "Federated setup:",
        f"clients={args.num_clients}",
        f"sampled_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"local_epochs={args.local_epochs}",
        f"dirichlet_alpha={args.dirichlet_alpha}",
        f"device={device}",
    )
    print(
        "Client data sizes:",
        f"min={min(client_sizes)} max={max(client_sizes)} avg={sum(client_sizes) / len(client_sizes):.1f}",
    )

    global_model = build_model(args).to(device)

    for round_idx in range(1, args.rounds + 1):
        selected_clients = random.sample(range(args.num_clients), args.clients_per_round)
        if round_idx == 1:
            print_first_round_distribution(train_targets, selected_clients, client_indices)

        global_state = clone_state_dict_to_cpu(global_model.state_dict())
        local_states = []
        sample_counts = []

        for client_id in selected_clients:
            local_seed = args.seed + round_idx * 1000 + client_id
            local_state, sample_count = train_local_model(
                global_state,
                client_loaders[client_id],
                client_sizes[client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        next_state = aggregate_states(local_states, sample_counts)
        global_model.load_state_dict(next_state)

        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(
            f"Round {round_idx:03d}:",
            f"selected_clients={selected_clients}",
            f"test_loss={test_loss:.4f}",
            f"test_acc={test_acc:.4f}",
        )

    return global_model


def main():
    args = parse_args()
    run_federated_learning(args)


if __name__ == "__main__":
    main()
